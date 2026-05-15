"""
app/services/nlp/skill_dictionary.py
──────────────────────────────────────
Master skill taxonomy used for extraction and matching.
Organized by category for targeted extraction and scoring.
"""

SKILL_DICTIONARY: dict[str, list[str]] = {
    "programming_languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#",
        "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R",
        "MATLAB", "Dart", "Elixir", "Haskell", "Lua", "Perl", "Shell",
        "Bash", "PowerShell", "SQL", "PL/SQL", "Solidity", "Assembly",
    ],
    "frontend_frameworks": [
        "React", "React.js", "Next.js", "Vue", "Vue.js", "Nuxt.js",
        "Angular", "Svelte", "SvelteKit", "Gatsby", "Remix", "Astro",
        "Ember.js", "Backbone.js", "jQuery", "Bootstrap", "Tailwind",
        "Tailwind CSS", "Material UI", "Ant Design", "Chakra UI",
        "Styled Components", "SASS", "SCSS", "Less", "HTML", "CSS",
        "WebAssembly", "WASM", "Three.js", "D3.js", "Chart.js",
    ],
    "backend_frameworks": [
        "Node.js", "Express", "Express.js", "NestJS", "Fastify",
        "Django", "Flask", "FastAPI", "Spring", "Spring Boot",
        "Laravel", "Symfony", "Ruby on Rails", "Rails", "ASP.NET",
        ".NET Core", "Gin", "Echo", "Fiber", "Phoenix", "Actix",
        "GraphQL", "REST", "gRPC", "WebSockets",
    ],
    "databases": [
        "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis",
        "Elasticsearch", "Cassandra", "DynamoDB", "Firebase",
        "Firestore", "Supabase", "CockroachDB", "MariaDB",
        "Oracle", "MS SQL Server", "Neo4j", "InfluxDB", "Couchbase",
        "Prisma", "Mongoose", "Sequelize", "SQLAlchemy", "TypeORM",
    ],
    "cloud_platforms": [
        "AWS", "Amazon Web Services", "GCP", "Google Cloud", "Azure",
        "Microsoft Azure", "Heroku", "DigitalOcean", "Vercel",
        "Netlify", "Cloudflare", "Linode", "Vultr", "IBM Cloud",
        "Oracle Cloud", "Alibaba Cloud",
    ],
    "devops_tools": [
        "Docker", "Kubernetes", "K8s", "Terraform", "Ansible",
        "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
        "Travis CI", "ArgoCD", "Helm", "Prometheus", "Grafana",
        "Nginx", "Apache", "HAProxy", "Istio", "Linkerd",
        "Vagrant", "Packer", "Pulumi", "Chef", "Puppet",
    ],
    "ai_ml_tools": [
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "sklearn",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
        "OpenCV", "NLTK", "spaCy", "Hugging Face", "Transformers",
        "BERT", "GPT", "LangChain", "LlamaIndex", "FAISS",
        "XGBoost", "LightGBM", "CatBoost", "MLflow", "Weights & Biases",
        "Jupyter", "Colab", "CUDA", "OpenAI API",
    ],
    "mobile_development": [
        "React Native", "Flutter", "iOS", "Android", "SwiftUI",
        "Jetpack Compose", "Xamarin", "Ionic", "Capacitor",
        "Expo", "Kotlin Multiplatform",
    ],
    "version_control_collaboration": [
        "Git", "GitHub", "GitLab", "Bitbucket", "SVN",
        "Jira", "Confluence", "Notion", "Slack", "Linear",
    ],
    "testing": [
        "Jest", "Pytest", "Mocha", "Cypress", "Playwright",
        "Selenium", "JUnit", "TestNG", "Vitest", "Jasmine",
        "Postman", "Insomnia", "k6", "Locust",
    ],
    "security": [
        "OAuth", "OAuth2", "JWT", "SSL", "TLS", "HTTPS",
        "OWASP", "Penetration Testing", "Cybersecurity",
        "Cryptography", "Zero Trust", "SIEM",
    ],
    "blockchain_web3": [
        "Solidity", "Web3.js", "Ethers.js", "Hardhat", "Truffle",
        "IPFS", "NFT", "Smart Contracts", "DeFi", "Ethereum",
        "Polygon", "Solana",
    ],
    "design_tools": [
        "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator",
        "InVision", "Zeplin", "Framer", "Canva",
    ],
    "soft_skills": [
        "Project Management", "Agile", "Scrum", "Kanban",
        "Communication", "Leadership", "Team Management",
        "Problem Solving", "Critical Thinking",
    ],
}

# ── Flat set for fast O(1) lookup ──────────────────────────────────────────────
ALL_SKILLS: set[str] = {
    skill.lower()
    for skills in SKILL_DICTIONARY.values()
    for skill in skills
}

# ── Reverse map: lowercase → canonical form ────────────────────────────────────
SKILL_CANONICAL: dict[str, str] = {
    skill.lower(): skill
    for skills in SKILL_DICTIONARY.values()
    for skill in skills
}

# ── Aliases / common abbreviations ────────────────────────────────────────────
SKILL_ALIASES: dict[str, str] = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "rb": "Ruby",
    "k8": "Kubernetes",
    "k8s": "Kubernetes",
    "tf": "TensorFlow",
    "pt": "PyTorch",
    "sk-learn": "Scikit-learn",
    "mongo": "MongoDB",
    "postgres": "PostgreSQL",
    "pg": "PostgreSQL",
    "mssql": "MS SQL Server",
    "gcp": "Google Cloud",
    "aws": "Amazon Web Services",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "oop": "Object-Oriented Programming",
    "api": "REST",
    "rest api": "REST",
    "react.js": "React",
    "vue.js": "Vue",
    "next": "Next.js",
    "nuxt": "Nuxt.js",
    "express.js": "Express",
    "node": "Node.js",
    "nodejs": "Node.js",
    "reactjs": "React",
    "vuejs": "Vue",
    "angular js": "Angular",
}

# ── Experience indicator phrases ───────────────────────────────────────────────
EXPERIENCE_PHRASES = {
    "senior": 3,
    "lead": 4,
    "principal": 5,
    "staff": 5,
    "architect": 6,
    "expert": 4,
    "junior": 1,
    "entry level": 0,
    "mid level": 2,
    "mid-level": 2,
    "intermediate": 2,
    "beginner": 0,
}

# ── Complexity indicator keywords ──────────────────────────────────────────────
COMPLEXITY_KEYWORDS = {
    "advanced": [
        "microservices", "distributed", "scalable", "high-availability",
        "kubernetes", "machine learning", "ai", "blockchain", "real-time",
        "millions of users", "enterprise", "cloud-native", "ci/cd pipeline",
        "production", "deployed", "architecture",
    ],
    "medium": [
        "rest api", "database", "authentication", "dashboard", "crud",
        "full stack", "fullstack", "responsive", "integration", "deployment",
        "docker", "testing", "api integration",
    ],
    "simple": [
        "landing page", "portfolio website", "todo", "calculator",
        "static website", "tutorial", "beginner", "basic", "simple",
        "homework", "exercise", "practice",
    ],
}