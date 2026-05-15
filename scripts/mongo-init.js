// scripts/mongo-init.js
// Runs once when MongoDB container starts for the first time

db = db.getSiblingDB("masar_ai");

// Create collections with schema validation
db.createCollection("freelancer_profiles");
db.createCollection("portfolio_analyses");
db.createCollection("skill_scores");
db.createCollection("match_results");
db.createCollection("ranking_scores");
db.createCollection("embeddings");

print("✅ Masar AI MongoDB collections initialized");