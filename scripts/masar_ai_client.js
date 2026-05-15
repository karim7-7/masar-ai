/**
 * scripts/masar_ai_client.js
 * ───────────────────────────
 * Node.js SDK / helper for calling the Masar AI microservice.
 * Drop this file into your Node.js backend and import it where needed.
 *
 * Usage:
 *   const masarAI = require('./masar_ai_client');
 *   const result = await masarAI.analyzePortfolio({ freelancerId: '...', portfolioText: '...' });
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

// ── Config ─────────────────────────────────────────────────────────────────────
const AI_SERVICE_URL = process.env.MASAR_AI_URL || 'http://localhost:8000';
const TIMEOUT_MS = 120_000;  // 2 minutes for heavy analysis

const client = axios.create({
  baseURL: AI_SERVICE_URL,
  timeout: TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json' },
});

// ── Error Wrapper ─────────────────────────────────────────────────────────────
function handleError(error, operation) {
  if (error.response) {
    const msg = `Masar AI ${operation} failed: ${error.response.status} — ${JSON.stringify(error.response.data)}`;
    throw new Error(msg);
  } else if (error.request) {
    throw new Error(`Masar AI service unreachable at ${AI_SERVICE_URL}`);
  }
  throw error;
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. HEALTH CHECK
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Check if the AI service is healthy.
 * @returns {{ status, version, services }}
 */
async function checkHealth() {
  try {
    const { data } = await client.get('/health');
    return data;
  } catch (error) {
    handleError(error, 'health check');
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. PORTFOLIO ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Analyze a freelancer's portfolio.
 *
 * @param {Object} params
 * @param {string} params.freelancerId       - MongoDB freelancer _id
 * @param {string} [params.portfolioText]    - Raw text from profile
 * @param {string} [params.githubUrl]        - GitHub profile/repo URL
 * @param {string} [params.pdfFilePath]      - Local path to CV PDF
 * @param {string} [params.pdfBase64]        - Base64-encoded PDF (alternative)
 * @returns {Promise<PortfolioAnalysisResult>}
 */
async function analyzePortfolio({ freelancerId, portfolioText, githubUrl, pdfFilePath, pdfBase64 }) {
  const body = { freelancer_id: freelancerId };

  if (portfolioText)   body.portfolio_text = portfolioText;
  if (githubUrl)       body.github_url = githubUrl;

  // Convert local PDF file to base64
  if (pdfFilePath && !pdfBase64) {
    const fileBuffer = fs.readFileSync(pdfFilePath);
    pdfBase64 = fileBuffer.toString('base64');
  }
  if (pdfBase64) body.pdf_base64 = pdfBase64;

  if (!portfolioText && !githubUrl && !pdfBase64) {
    throw new Error('analyzePortfolio: provide at least one of portfolioText, githubUrl, or pdfFilePath');
  }

  try {
    const { data } = await client.post('/analyze-portfolio', body);
    return data;
  } catch (error) {
    handleError(error, 'analyzePortfolio');
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. GENERATE EMBEDDING
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Generate and store an embedding for a freelancer or project.
 *
 * @param {Object} params
 * @param {string} params.entityId    - freelancer or project ID
 * @param {'freelancer'|'project'} params.entityType
 * @param {string} params.text        - Rich description text for embedding
 */
async function generateEmbedding({ entityId, entityType, text }) {
  try {
    const { data } = await client.post('/generate-embedding', {
      entity_id: entityId,
      entity_type: entityType,
      text,
    });
    return data;
  } catch (error) {
    handleError(error, 'generateEmbedding');
  }
}

/**
 * Helper: build embedding text from a freelancer profile document (MongoDB doc).
 */
function buildFreelancerEmbeddingText(freelancer) {
  const parts = [];
  if (freelancer.bio)             parts.push(freelancer.bio);
  if (freelancer.skills?.length)  parts.push(`Skills: ${freelancer.skills.map(s => s.name || s).join(', ')}`);
  if (freelancer.experienceLevel) parts.push(`Experience level: ${freelancer.experienceLevel}`);
  if (freelancer.yearsOfExperience) parts.push(`Years of experience: ${freelancer.yearsOfExperience}`);
  return parts.join(' | ');
}

/**
 * Helper: build embedding text from a project document (MongoDB doc).
 */
function buildProjectEmbeddingText(project) {
  const parts = [];
  if (project.title)          parts.push(project.title);
  if (project.description)    parts.push(project.description);
  if (project.requiredSkills?.length) parts.push(`Required: ${project.requiredSkills.join(', ')}`);
  if (project.complexity)     parts.push(`Complexity: ${project.complexity}`);
  return parts.join(' | ');
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. MATCH PROJECT
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Find best-matching freelancers for a project.
 *
 * @param {Object} project
 * @param {string} project.projectId
 * @param {string} project.title
 * @param {string} project.description
 * @param {string[]} [project.requiredSkills]
 * @param {'Simple'|'Medium'|'Advanced'} [project.complexity]
 * @param {'Beginner'|'Intermediate'|'Expert'} [project.experienceRequired]
 * @param {number} [topK=10]
 * @returns {Promise<MatchResult>}
 */
async function matchProject(project, topK = 10) {
  const body = {
    project_id: project.projectId,
    title: project.title,
    description: project.description,
    required_skills: project.requiredSkills || [],
    complexity: project.complexity,
    experience_required: project.experienceRequired,
    budget_range: project.budgetRange,
  };

  try {
    const { data } = await client.post(`/match-project?top_k=${topK}`, body);
    return data;
  } catch (error) {
    handleError(error, 'matchProject');
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. RANK FREELANCER
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Compute ranking score for a freelancer.
 *
 * @param {Object} params
 * @param {string} params.freelancerId
 * @param {number} params.skillRelevanceRaw    - 0-100
 * @param {number} params.portfolioScore       - 0-100 (from analyzePortfolio)
 * @param {number} params.avgClientRating      - 0-5 stars
 * @param {number} params.responseTimeHours    - Average first response hours
 * @param {number} params.completedProjects    - Total jobs done
 * @param {'Beginner'|'Intermediate'|'Expert'} params.experienceLevel
 * @param {boolean} [params.isSpamSuspected]
 */
async function rankFreelancer({
  freelancerId,
  skillRelevanceRaw,
  portfolioScore,
  avgClientRating,
  responseTimeHours,
  completedProjects,
  experienceLevel,
  isSpamSuspected = false,
}) {
  const body = {
    freelancer_id: freelancerId,
    skill_relevance_raw: skillRelevanceRaw,
    portfolio_score: portfolioScore,
    avg_client_rating: avgClientRating,
    response_time_hours: responseTimeHours,
    completed_projects: completedProjects,
    experience_level: experienceLevel,
    is_spam_suspected: isSpamSuspected,
  };

  try {
    const { data } = await client.post('/rank-freelancer', body);
    return data;
  } catch (error) {
    handleError(error, 'rankFreelancer');
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// WORKFLOW HELPERS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Full onboarding workflow:
 * Analyze portfolio → Generate embedding → Rank freelancer
 * Call this when a freelancer submits or updates their profile.
 *
 * @param {Object} freelancerDoc - MongoDB freelancer document
 * @param {Object} [options]
 * @param {string} [options.pdfBase64]    - Base64 CV if uploaded
 * @param {string} [options.githubUrl]    - GitHub URL
 */
async function onboardFreelancer(freelancerDoc, options = {}) {
  const freelancerId = freelancerDoc._id.toString();
  const results = {};

  console.log(`[Masar AI] Onboarding freelancer ${freelancerId}...`);

  // Step 1: Analyze portfolio
  try {
    const analysis = await analyzePortfolio({
      freelancerId,
      portfolioText: freelancerDoc.bio || freelancerDoc.description,
      githubUrl: options.githubUrl || freelancerDoc.githubUrl,
      pdfBase64: options.pdfBase64,
    });
    results.analysis = analysis;
    console.log(`[Masar AI] Analysis done — score: ${analysis.portfolio_score}`);
  } catch (err) {
    console.error('[Masar AI] Portfolio analysis failed:', err.message);
  }

  // Step 2: Generate embedding
  try {
    const embeddingText = buildFreelancerEmbeddingText({
      ...freelancerDoc,
      ...results.analysis,
    });
    const embedding = await generateEmbedding({
      entityId: freelancerId,
      entityType: 'freelancer',
      text: embeddingText,
    });
    results.embedding = embedding;
    console.log(`[Masar AI] Embedding stored — FAISS id: ${embedding.faiss_index_id}`);
  } catch (err) {
    console.error('[Masar AI] Embedding generation failed:', err.message);
  }

  return results;
}

/**
 * Full project matching workflow:
 * Generate project embedding → Match freelancers → Return ranked list
 *
 * @param {Object} projectDoc - MongoDB project document
 */
async function findFreelancersForProject(projectDoc, topK = 10) {
  const projectId = projectDoc._id.toString();

  // Step 1: Generate project embedding
  const embeddingText = buildProjectEmbeddingText(projectDoc);
  await generateEmbedding({
    entityId: projectId,
    entityType: 'project',
    text: embeddingText,
  });

  // Step 2: Match
  const matches = await matchProject({
    projectId,
    title: projectDoc.title,
    description: projectDoc.description,
    requiredSkills: projectDoc.requiredSkills || [],
    complexity: projectDoc.complexity,
    experienceRequired: projectDoc.experienceRequired,
  }, topK);

  return matches;
}

// ── Exports ───────────────────────────────────────────────────────────────────
module.exports = {
  checkHealth,
  analyzePortfolio,
  generateEmbedding,
  matchProject,
  rankFreelancer,
  onboardFreelancer,
  findFreelancersForProject,
  buildFreelancerEmbeddingText,
  buildProjectEmbeddingText,
};