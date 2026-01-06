const express = require("express");
const path = require("path");
const fs = require("fs");
const sqlite3 = require("sqlite3").verbose();
const multer = require("multer");
const dotenv = require("dotenv");
const fetch = require("node-fetch"); // direct HTTP calls to OpenAI

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3040;

const DB_PATH = path.join(process.cwd(), "data", "photo_index.db");
const IMAGES_ROOT = path.join(process.cwd(), "data", "images_root");

// Ensure images_root exists
if (!fs.existsSync(IMAGES_ROOT)) {
  fs.mkdirSync(IMAGES_ROOT, { recursive: true });
}

// ---------- DB SETUP (sqlite3, async) ----------

if (!fs.existsSync(DB_PATH)) {
  fs.writeFileSync(DB_PATH, "");
}

const db = new sqlite3.Database(DB_PATH);

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS photos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_path TEXT UNIQUE,
      project TEXT,
      shot_date TEXT,
      description TEXT,
      embedding BLOB,
      bookmarked INTEGER DEFAULT 0
    );
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS photo_passes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      photo_id INTEGER NOT NULL,
      pass_label TEXT,
      focus_prompt TEXT,
      description TEXT,
      embedding BLOB,
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
    );
  `);

  // Migration: Add bookmarked column to existing databases
  db.all("PRAGMA table_info(photos)", [], (err, columns) => {
    if (err) {
      console.error("Error checking photos table schema:", err);
      return;
    }
    const hasBookmarked = columns.some(col => col.name === 'bookmarked');
    if (!hasBookmarked) {
      db.run("ALTER TABLE photos ADD COLUMN bookmarked INTEGER DEFAULT 0", (err) => {
        if (err) {
          console.error("Error adding bookmarked column:", err);
        } else {
          console.log("✓ Added bookmarked column to existing photos table");
        }
      });
    }
  });
});

// Project summary helper
function getProjectsSummary() {
  return new Promise((resolve, reject) => {
    db.all(
      `
      SELECT p.project, COUNT(*) as count, MIN(shot_date) as earliest, MAX(shot_date) as latest
      , COALESCE(SUM(pc.pass_count), 0) as pass_count
      FROM photos p
      LEFT JOIN (
        SELECT photo_id, COUNT(*) as pass_count
        FROM photo_passes
        GROUP BY photo_id
      ) pc ON pc.photo_id = p.id
      GROUP BY p.project
      ORDER BY p.project COLLATE NOCASE
    `,
      [],
      (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      }
    );
  });
}

// Insert helper
function insertPhoto(record) {
  return new Promise((resolve, reject) => {
    db.run(
      `
      INSERT OR IGNORE INTO photos
      (file_path, project, shot_date, description, embedding)
      VALUES (?, ?, ?, ?, ?)
    `,
      [
        record.file_path,
        record.project,
        record.shot_date,
        record.description,
        record.embedding,
      ],
      function (err) {
        if (err) reject(err);
        else resolve();
      }
    );
  });
}

// Fetch helper (with optional filters)
function getAllPhotos(projectFilter, dateFilter) {
  return new Promise((resolve, reject) => {
    let sql = "SELECT * FROM photos WHERE 1=1";
    const params = [];

    if (projectFilter) {
      sql += " AND project = ?";
      params.push(projectFilter);
    }
    if (dateFilter) {
      sql += " AND shot_date = ?";
      params.push(dateFilter);
    }

    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

function getPhotoById(id) {
  return new Promise((resolve, reject) => {
    db.get("SELECT * FROM photos WHERE id = ?", [id], (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
}

function getPassesForPhoto(photoId) {
  return new Promise((resolve, reject) => {
    db.all(
      `SELECT * FROM photo_passes WHERE photo_id = ? ORDER BY id DESC`,
      [photoId],
      (err, rows) => {
        if (err) reject(err);
        else resolve(rows || []);
      }
    );
  });
}

// ---------- Helpers ----------

function todayString() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function fileToBase64(filePath) {
  const data = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase().replace(".", "") || "jpeg";
  return `data:image/${ext};base64,` + data.toString("base64");
}

// Call OpenAI chat completions with vision via HTTP
async function describeImageWithAI(imagePath, focusPrompt = "") {
  const base64 = fileToBase64(imagePath);

  const focusInstruction = focusPrompt
    ? `Focus on: ${focusPrompt}.`
    : "";

  const body = {
    model: "gpt-4.1-mini", // vision-capable model
    messages: [
      {
        role: "system",
        content:
          "You are a construction progress inspector who writes terse, technical captions. Identify trade (e.g., concrete, structural steel, roofing, MEP rough-in), primary activity, major equipment, materials, location context, and stage of completion. If a user provided focus, emphasize those details.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text:
              `Provide a single sentence: start with trade + activity, then materials/equipment, and any QA/QC notes (formwork status, rebar spacing, welds, waterproofing, PPE). ${focusInstruction}`.trim(),
          },
          {
            type: "image_url",
            image_url: { url: base64 },
          },
        ],
      },
    ],
    max_tokens: 80,
  };

  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("OpenAI vision error:", resp.status, errText);
    throw new Error("OpenAI vision request failed");
  }

  const json = await resp.json();
  const text = json.choices?.[0]?.message?.content?.trim();
  return text || "Construction site photo.";
}

// Call OpenAI embeddings via HTTP
async function getEmbedding(text) {
  const body = {
    model: "text-embedding-3-small",
    input: text,
  };

  const resp = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("OpenAI embeddings error:", resp.status, errText);
    throw new Error("OpenAI embeddings request failed");
  }

  const json = await resp.json();
  return json.data?.[0]?.embedding;
}

async function embedText(text) {
  const vector = await getEmbedding(text);
  const floatArray = new Float32Array(vector);
  return Buffer.from(floatArray.buffer);
}

async function embedQuery(text) {
  return getEmbedding(text);
}

function blobToVector(blob) {
  if (!blob) return null;
  const buffer = Buffer.from(blob);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function insertPhotoPass(record) {
  return new Promise((resolve, reject) => {
    db.run(
      `
      INSERT INTO photo_passes
      (photo_id, pass_label, focus_prompt, description, embedding)
      VALUES (?, ?, ?, ?, ?)
    `,
      [
        record.photo_id,
        record.pass_label,
        record.focus_prompt,
        record.description,
        record.embedding,
      ],
      function (err) {
        if (err) reject(err);
        else resolve();
      }
    );
  });
}

function deletePhotoRecord(id) {
  return new Promise((resolve, reject) => {
    db.run("DELETE FROM photos WHERE id = ?", [id], function (err) {
      if (err) reject(err);
      else resolve();
    });
  });
}

function deletePassesForPhoto(photoId) {
  return new Promise((resolve, reject) => {
    db.run(
      "DELETE FROM photo_passes WHERE photo_id = ?",
      [photoId],
      function (err) {
        if (err) reject(err);
        else resolve();
      }
    );
  });
}

function toggleBookmark(photoId) {
  return new Promise((resolve, reject) => {
    db.run(
      "UPDATE photos SET bookmarked = NOT bookmarked WHERE id = ?",
      [photoId],
      function (err) {
        if (err) reject(err);
        else {
          db.get(
            "SELECT bookmarked FROM photos WHERE id = ?",
            [photoId],
            (err, row) => {
              if (err) reject(err);
              else resolve({ bookmarked: row?.bookmarked || 0 });
            }
          );
        }
      }
    );
  });
}

function getBookmarkedPhotos() {
  return new Promise((resolve, reject) => {
    db.all(
      "SELECT * FROM photos WHERE bookmarked = 1 ORDER BY shot_date DESC, id DESC",
      [],
      (err, rows) => {
        if (err) reject(err);
        else resolve(rows || []);
      }
    );
  });
}

// ---------- Multer upload config ----------

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const projectRaw = req.body.project || "UnknownProject";
    const project = projectRaw.replace(/[^a-zA-Z0-9_-]/g, "_");
    const dateFolder = req.body.shot_date || todayString();

    const dest = path.join(IMAGES_ROOT, project, dateFolder);
    fs.mkdirSync(dest, { recursive: true });
    cb(null, dest);
  },
  filename: function (req, file, cb) {
    const ts = Date.now();
    const ext = path.extname(file.originalname) || ".jpg";
    const cleanName = file.originalname.replace(/\s+/g, "_");
    cb(null, `${ts}-${cleanName}${ext}`);
  },
});

const upload = multer({ storage });

// ---------- Middleware / static ----------

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use("/images", express.static(IMAGES_ROOT));

// ---------- Reindex job store ----------

const reindexJobs = new Map();

function makeJobId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// ---------- Routes ----------

// Upload & index route
app.post("/upload", upload.array("photos", 300), async (req, res) => {
  const projectRaw = req.body.project || "UnknownProject";
  const project = projectRaw.replace(/[^a-zA-Z0-9_-]/g, "_");
  const shot_date = req.body.shot_date || todayString();
  const files = req.files || [];

  if (!files.length) {
    return res.status(400).send("No files uploaded.");
  }

  for (const file of files) {
    try {
      const absPath = file.path;
      const relativePath = path.relative(IMAGES_ROOT, absPath);

      const description = await describeImageWithAI(absPath);
      const embeddingText = `${description} | Project:${project} | Shot:${shot_date}`;
      const embedding = await embedText(embeddingText);

      await insertPhoto({
        file_path: relativePath,
        project,
        shot_date,
        description,
        embedding,
      });

      console.log(`Indexed uploaded photo: ${relativePath}`);
    } catch (err) {
      console.error(
        "Error indexing uploaded file:",
        file.filename,
        err.message
      );
    }
  }

  res.send(`
    <html>
      <body>
        <p>Uploaded and indexed ${files.length} photo(s) for project <strong>${project}</strong>.</p>
        <p><a href="/">Back to search</a></p>
      </body>
    </html>
  `);
});

// Search route
app.get("/search", async (req, res) => {
  const query = req.query.q;
  const limit = parseInt(req.query.limit || "20", 10);
  const projectFilter = req.query.project || "";
  const dateFilter = req.query.shot_date || "";

  if (!query) {
    return res.status(400).json({ error: "Missing 'q' query parameter" });
  }

  try {
    const queryEmbedding = await embedQuery(query);
    const qVec = Float32Array.from(queryEmbedding);

    const rows = await getAllPhotos(projectFilter, dateFilter);

    const scored = await Promise.all(
      rows.map(async (row) => {
        const embVec = blobToVector(row.embedding);
        if (!embVec) return null;

        const passes = await getPassesForPhoto(row.id);
        const passVectors = passes
          .map((p) => ({
            ...p,
            vector: blobToVector(p.embedding),
          }))
          .filter((p) => !!p.vector);

        let bestScore = cosineSimilarity(qVec, embVec);
        let bestPassLabel = "Base";
        let bestDescription = row.description;

        for (const p of passVectors) {
          const candidate = cosineSimilarity(qVec, p.vector);
          if (candidate > bestScore) {
            bestScore = candidate;
            bestPassLabel = p.pass_label || "Focused pass";
            bestDescription = p.description || bestDescription;
          }
        }

        const contextPasses = [
          {
            id: `base-${row.id}`,
            label: "Base",
            description: row.description,
            created_at: row.shot_date,
          },
          ...passes.map((p) => ({
            id: p.id,
            label: p.pass_label || "Focused pass",
            description: p.description,
            created_at: p.created_at,
          })),
        ];

        return {
          ...row,
          score: bestScore,
          bestPassLabel,
          bestDescription,
          passes: contextPasses,
        };
      })
    );

    const filteredScored = scored.filter(Boolean);

    // Sort best → worst
    filteredScored.sort((a, b) => b.score - a.score);

    // 🔹 Only keep “good enough” matches
    const MIN_SCORE = 0.3; // tweak this to adjust strictness
    const filtered = filteredScored.filter((row) => row.score >= MIN_SCORE);

    const top = filtered.slice(0, limit).map((row) => ({
      id: row.id,
      file_path: row.file_path,
      project: row.project,
      shot_date: row.shot_date,
      description: row.bestDescription,
      best_pass_label: row.bestPassLabel,
      score: row.score,
      passes: row.passes,
      image_url: `/images/${row.file_path.replace(/\\/g, "/")}`,
    }));

    res.json({ query, results: top });
  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Project summary list
app.get("/projects", async (req, res) => {
  try {
    const projects = await getProjectsSummary();
    res.json({ projects });
  } catch (err) {
    console.error("Projects list error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Photos by project
app.get("/projects/:project/photos", async (req, res) => {
  const project = decodeURIComponent(req.params.project || "");
  if (!project) {
    return res.status(400).json({ error: "Missing project" });
  }

  try {
    const rows = await getAllPhotos(project, "");
    const withPasses = await Promise.all(
      rows.map(async (row) => {
        const passes = await getPassesForPhoto(row.id);
        return {
          ...row,
          passes: [
            {
              id: `base-${row.id}`,
              label: "Base",
              description: row.description,
              created_at: row.shot_date,
            },
            ...passes.map((p) => ({
              id: p.id,
              label: p.pass_label || "Focused pass",
              description: p.description,
              created_at: p.created_at,
            })),
          ],
        };
      })
    );

    const ordered = withPasses
      .map((row) => ({
        ...row,
        image_url: `/images/${row.file_path.replace(/\\/g, "/")}`,
      }))
      .sort((a, b) => {
        if (a.shot_date === b.shot_date) return b.id - a.id;
        return (b.shot_date || "").localeCompare(a.shot_date || "");
      });

    res.json({ project, photos: ordered });
  } catch (err) {
    console.error("Project photos error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Delete photo
app.delete("/photos/:id", async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!id) return res.status(400).json({ error: "Invalid id" });

  try {
    const row = await getPhotoById(id);
    if (!row) return res.status(404).json({ error: "Not found" });

    const fullPath = path.join(IMAGES_ROOT, row.file_path);
    if (fs.existsSync(fullPath)) {
      try {
        fs.unlinkSync(fullPath);
      } catch (err) {
        console.warn("Could not delete file", fullPath, err.message);
      }
    }

    await deletePassesForPhoto(id);
    await deletePhotoRecord(id);
    res.json({ ok: true });
  } catch (err) {
    console.error("Delete photo error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Toggle bookmark
app.post("/photos/:id/bookmark", async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!id) return res.status(400).json({ error: "Invalid id" });

  try {
    const row = await getPhotoById(id);
    if (!row) return res.status(404).json({ error: "Not found" });

    const result = await toggleBookmark(id);
    res.json({ ok: true, bookmarked: result.bookmarked });
  } catch (err) {
    console.error("Toggle bookmark error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Get bookmarked photos
app.get("/bookmarked", async (req, res) => {
  try {
    const rows = await getBookmarkedPhotos();
    const withPasses = await Promise.all(
      rows.map(async (row) => {
        const passes = await getPassesForPhoto(row.id);
        return {
          ...row,
          passes: [
            {
              id: `base-${row.id}`,
              label: "Base",
              description: row.description,
              created_at: row.shot_date,
            },
            ...passes.map((p) => ({
              id: p.id,
              label: p.pass_label || "Focused pass",
              description: p.description,
              created_at: p.created_at,
            })),
          ],
        };
      })
    );

    const photos = withPasses.map((row) => ({
      ...row,
      image_url: `/images/${row.file_path.replace(/\\/g, "/")}`,
    }));

    res.json({ photos });
  } catch (err) {
    console.error("Get bookmarked photos error:", err);
    res.status(500).json({ error: "Internal error" });
  }
});

// Kick off focused re-index job
app.post("/reindex", async (req, res) => {
  const photoIds = Array.isArray(req.body.photoIds)
    ? req.body.photoIds.map((p) => parseInt(p, 10)).filter(Boolean)
    : [];
  const focus = (req.body.focus || "").toString().trim();
  const label = (req.body.label || "Focused pass").toString().trim() || "Focused pass";

  if (!photoIds.length) {
    return res.status(400).json({ error: "No photos selected" });
  }

  const jobId = makeJobId();
  const jobState = {
    id: jobId,
    total: photoIds.length,
    done: 0,
    status: "running",
    errors: [],
    results: [],
    started_at: new Date().toISOString(),
  };

  reindexJobs.set(jobId, jobState);

  (async () => {
    for (const pid of photoIds) {
      try {
        const row = await getPhotoById(pid);
        if (!row) {
          jobState.errors.push({ photoId: pid, message: "Not found" });
          continue;
        }

        const absPath = path.join(IMAGES_ROOT, row.file_path);
        const description = await describeImageWithAI(absPath, focus);
        const embeddingText = `${description} | Project:${row.project} | Shot:${row.shot_date} | Focus:${focus}`;
        const embedding = await embedText(embeddingText);

        await insertPhotoPass({
          photo_id: pid,
          pass_label: label,
          focus_prompt: focus,
          description,
          embedding,
        });

        jobState.results.push({ photoId: pid, label });
      } catch (err) {
        jobState.errors.push({ photoId: pid, message: err.message || "Error" });
      } finally {
        jobState.done += 1;
      }
    }

    jobState.status = "completed";
    jobState.completed_at = new Date().toISOString();
  })();

  res.json({ jobId });
});

// Poll job progress
app.get("/reindex/:jobId", (req, res) => {
  const job = reindexJobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

// Simple UI
app.get("/", (req, res) => {
  res.send(`
    <!doctype html>
    <html>
      <head>
        <title>Job Photo Search</title>
        <style>
          :root {
            color-scheme: light;
            --accent: #1d4ed8;
            --accent-dark: #1e3a8a;
          }
          body {
            font-family: system-ui, sans-serif;
            margin: 0;
            padding: 24px;
            background: radial-gradient(circle at 10% 20%, #f5f9ff, #f8f8f8 45%);
            color: #1f2937;
          }
          h1 { margin-bottom: 0.25rem; letter-spacing: -0.02em; }
          h2 { margin-top: 1rem; }
          p.lead { margin-top: 0; color: #4b5563; }
          label { display: block; margin-top: 10px; font-weight: 600; }
          input[type="text"], input[type="date"], input[type="file"], select {
            padding: 10px;
            width: 100%;
            max-width: 420px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            margin-top: 4px;
            font-size: 14px;
          }
          button {
            padding: 10px 16px;
            margin-top: 12px;
            cursor: pointer;
            background: linear-gradient(135deg, var(--accent-dark), var(--accent));
            border: none;
            color: white;
            border-radius: 10px;
            font-weight: 700;
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
          }
          button:hover { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(30, 58, 138, 0.28); }
          button:active { transform: translateY(0); box-shadow: 0 6px 12px rgba(30, 58, 138, 0.2); }
          .shell { max-width: 1100px; margin: 0 auto; }
          .section { border: 1px solid #e5e7eb; padding: 18px; border-radius: 14px; margin-bottom: 18px; background: white; box-shadow: 0 12px 40px rgba(15, 23, 42, 0.05); }
          #results, #projectPhotos { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 20px; }
          .photo-card { border: 1px solid #e5e7eb; padding: 10px; border-radius: 12px; background: #fdfefe; box-shadow: inset 0 0 0 1px #f3f4f6; transition: transform 0.12s ease, box-shadow 0.12s ease; }
          .photo-card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
          .photo-card img { max-width: 100%; display: block; border-radius: 10px; margin-bottom: 8px; cursor: pointer; }
          .meta { font-size: 13px; color: #4b5563; margin-top: 4px; line-height: 1.4; }
          .meta strong { color: #111827; }
          .score { font-size: 12px; color: #6b7280; }
          .actions { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 8px; }
          .actions .stack { display: flex; gap: 8px; flex-wrap: wrap; }
          .pill { display: inline-block; padding: 4px 10px; background: #dbeafe; color: #1e3a8a; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.01em; }
          .tab-bar { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
          .tab-btn { padding: 10px 14px; border-radius: 999px; border: 1px solid #e5e7eb; background: #f3f4f6; color: #111827; font-weight: 700; cursor: pointer; }
          .tab-btn.active { background: linear-gradient(135deg, var(--accent-dark), var(--accent)); color: white; border-color: transparent; box-shadow: 0 10px 30px rgba(30,58,138,0.25); }
          .tab-panel { display: none; }
          .tab-panel.active { display: block; }
          .status { margin-top: 10px; color: #374151; font-size: 14px; }
          .progress-shell { width: 100%; height: 12px; border-radius: 999px; background: #e5e7eb; overflow: hidden; margin-top: 8px; }
          .progress-bar { height: 100%; width: 0%; background: linear-gradient(135deg, #10b981, #22d3ee); transition: width 0.2s ease; }
          .inline-group { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
          .project-chip { padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 12px; background: #f8fafc; display: flex; justify-content: space-between; align-items: center; cursor: pointer; transition: box-shadow 0.12s ease; }
          .project-chip:hover { box-shadow: 0 8px 22px rgba(0,0,0,0.08); }
          .project-chip strong { color: #111827; }
          .soft { color: #6b7280; font-size: 13px; }
          .danger { background: linear-gradient(135deg, #ef4444, #f97316); box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25); }
          .lightbox { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.6); display: none; justify-content: center; align-items: center; padding: 20px; z-index: 20; }
          .lightbox.active { display: flex; }
          .lightbox-card { background: white; border-radius: 16px; padding: 16px; max-width: 90vw; max-height: 90vh; box-shadow: 0 20px 50px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 12px; }
          .lightbox-card img { max-height: 60vh; width: auto; object-fit: contain; border-radius: 12px; }
          .lightbox-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
          .lightbox-meta { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
          .lightbox-actions { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
          .ghost { background: transparent; color: #111827; border: 1px solid #e5e7eb; box-shadow: none; }
          .context-modal { position: fixed; inset: 0; background: rgba(17, 24, 39, 0.45); display: none; justify-content: center; align-items: center; padding: 20px; z-index: 30; }
          .context-modal.active { display: flex; }
          .context-card { background: #fff; border-radius: 14px; padding: 18px; max-width: 640px; width: 100%; box-shadow: 0 20px 60px rgba(0,0,0,0.22); max-height: 80vh; overflow: auto; }
          .context-list { display: grid; gap: 10px; margin-top: 10px; }
          .context-item { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; background: #f9fafb; }
          .context-item strong { display: inline-block; margin-bottom: 4px; color: #111827; }
          .reindex-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
          .checkbox-card { border: 1px solid #e5e7eb; padding: 10px; border-radius: 12px; background: #f9fafb; display: flex; gap: 8px; align-items: flex-start; }
          .bookmark-btn { background: transparent; border: 2px solid transparent; cursor: pointer; padding: 4px 8px; font-size: 20px; transition: all 0.2s ease; box-shadow: none; margin: 0; border-radius: 8px; }
          .bookmark-btn:hover { transform: scale(1.1); }
          .bookmark-btn:active { transform: scale(0.95); }
          .bookmark-btn.bookmarked { color: #1d4ed8; background: #dbeafe; border-color: #1d4ed8; }
          .bookmark-btn:not(.bookmarked) { color: #9ca3af; background: #f9fafb; border-color: #e5e7eb; }
          .bookmark-btn:not(.bookmarked):hover { background: #f3f4f6; border-color: #d1d5db; }
          #bookmarkedPhotos { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; margin-top: 20px; }
        </style>
      </head>
      <body>
        <div class="shell">
          <h1>Job Photo Search</h1>
          <p class="lead">Upload field photos, let AI label them, then search or browse by project.</p>

          <div class="tab-bar">
            <button class="tab-btn active" data-tab-target="searchTab">Search</button>
            <button class="tab-btn" data-tab-target="bookmarkedTab">Bookmarked</button>
            <button class="tab-btn" data-tab-target="uploadTab">Upload</button>
            <button class="tab-btn" data-tab-target="focusTab">Focused indexing</button>
            <button class="tab-btn" data-tab-target="projectsTab">Projects</button>
          </div>

          <div id="searchTab" class="tab-panel active section">
            <h2>Search photos</h2>
            <form id="searchForm">
              <label>
                Search description or concepts:
                <input type="text" id="q" name="q" placeholder="e.g. scaffolding, footing rebar, excavator digging" required />
              </label>
              <label>
                Filter by project (optional):
                <input type="text" id="projectFilter" name="project" placeholder="exact project name" />
              </label>
              <label>
                Filter by shot date (optional):
                <input type="date" id="shotDateFilter" name="shot_date" />
              </label>
              <button type="submit">Search</button>
            </form>
            <div id="results"></div>
          </div>

          <div id="bookmarkedTab" class="tab-panel section">
            <h2>Bookmarked photos</h2>
            <p class="soft">Your bookmarked photos appear here for quick access.</p>
            <div id="bookmarkedPhotos"></div>
          </div>

          <div id="uploadTab" class="tab-panel section">
            <h2>Upload photos</h2>
            <p class="soft">Choose an existing project or create a new one. We'll index the images as they're uploaded.</p>
            <div class="inline-group">
              <label>
                <input type="radio" name="projectMode" value="existing" checked /> Use existing project
              </label>
              <label>
                <input type="radio" name="projectMode" value="new" /> Create new project
              </label>
            </div>
            <form id="uploadForm" enctype="multipart/form-data">
              <label>
                Existing project:
                <select id="existingProject" name="existing_project"></select>
              </label>
              <label>
                New project name:
                <input type="text" id="newProject" name="new_project" placeholder="e.g. NorthRidgevilleHS" />
              </label>
              <label>
                Shot date (optional, defaults to today):
                <input type="date" id="shotDate" name="shot_date" />
              </label>
              <label>
                Photos:
                <input type="file" id="photos" name="photos" multiple accept="image/*" required />
              </label>
              <button type="submit" id="uploadBtn">Upload & Index</button>
              <div class="status" id="uploadStatus"></div>
              <div class="progress-shell"><div id="uploadProgress" class="progress-bar"></div></div>
            </form>
          </div>

          <div id="focusTab" class="tab-panel section">
            <h2>Focused re-index</h2>
            <p class="soft">Run an additional AI pass on selected photos with a custom focus prompt. Results are additive and won't replace the base indexing.</p>
            <label>
              Choose project(s):
              <div id="focusProjectList" class="inline-group"></div>
            </label>
            <div id="focusPhotos" class="reindex-grid"></div>
            <label>
              Focus prompt to emphasize:
              <textarea id="focusPrompt" rows="3" style="width:100%; max-width: 640px; padding: 10px; border-radius: 10px; border: 1px solid #d1d5db;"></textarea>
            </label>
            <label>
              Label for this pass (optional):
              <input type="text" id="focusLabel" placeholder="e.g. Interior finishes details" />
            </label>
            <button type="button" id="runFocusBtn">Run focused pass</button>
            <div class="status" id="focusStatus"></div>
            <div class="progress-shell"><div id="focusProgress" class="progress-bar"></div></div>
          </div>

          <div id="projectsTab" class="tab-panel section">
            <h2>Browse by project</h2>
            <p class="soft">See indexed projects, open images full-size, or delete unwanted shots.</p>
            <div id="projectsList" class="inline-group"></div>
            <div id="projectPhotos"></div>
          </div>
        </div>

        <div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="Image viewer">
          <div class="lightbox-card">
            <div class="lightbox-header">
              <div class="meta" id="lightboxTitle"></div>
              <div style="display: flex; gap: 8px; align-items: center;">
                <button type="button" class="bookmark-btn" id="lightboxBookmark" title="Bookmark this photo">🔖</button>
                <button type="button" class="ghost" id="closeLightbox">Close</button>
              </div>
            </div>
            <img id="lightboxImage" src="" alt="Expanded view" />
            <div class="lightbox-meta">
              <div class="meta" id="lightboxCaption"></div>
              <span class="score" id="lightboxCounter"></span>
            </div>
            <div class="lightbox-actions">
              <button type="button" class="ghost" id="prevLightbox">Previous</button>
              <button type="button" id="nextLightbox">Next</button>
            </div>
          </div>
        </div>

        <div id="contextModal" class="context-modal" role="dialog" aria-modal="true" aria-label="Pass contexts">
          <div class="context-card">
            <div class="lightbox-header" style="margin-bottom: 8px;">
              <div class="meta" id="contextTitle"></div>
              <button type="button" class="ghost" id="closeContext">Close</button>
            </div>
            <div id="contextList" class="context-list"></div>
          </div>
        </div>

        <script>
          const searchForm = document.getElementById('searchForm');
          const resultsDiv = document.getElementById('results');
          const bookmarkedPhotos = document.getElementById('bookmarkedPhotos');
          const uploadForm = document.getElementById('uploadForm');
          const uploadStatus = document.getElementById('uploadStatus');
          const uploadProgress = document.getElementById('uploadProgress');
          const uploadBtn = document.getElementById('uploadBtn');
          const projectsList = document.getElementById('projectsList');
          const projectPhotos = document.getElementById('projectPhotos');
          const existingProjectSelect = document.getElementById('existingProject');
          const focusProjectList = document.getElementById('focusProjectList');
          const focusPhotos = document.getElementById('focusPhotos');
          const focusPrompt = document.getElementById('focusPrompt');
          const focusLabel = document.getElementById('focusLabel');
          const focusStatus = document.getElementById('focusStatus');
          const focusProgress = document.getElementById('focusProgress');
          const runFocusBtn = document.getElementById('runFocusBtn');
          const lightbox = document.getElementById('lightbox');
          const lightboxImage = document.getElementById('lightboxImage');
          const lightboxTitle = document.getElementById('lightboxTitle');
          const lightboxCaption = document.getElementById('lightboxCaption');
          const lightboxCounter = document.getElementById('lightboxCounter');
          const lightboxBookmark = document.getElementById('lightboxBookmark');
          const closeLightboxBtn = document.getElementById('closeLightbox');
          const prevLightboxBtn = document.getElementById('prevLightbox');
          const nextLightboxBtn = document.getElementById('nextLightbox');
          const contextModal = document.getElementById('contextModal');
          const contextTitle = document.getElementById('contextTitle');
          const contextList = document.getElementById('contextList');
          const closeContextBtn = document.getElementById('closeContext');

          let lightboxItems = [];
          let lightboxIndex = 0;
          let latestSearchResults = [];
          let latestBookmarkedPhotos = [];
          let indexingTimer = null;
          focusPhotos.innerHTML = '<p class="meta">Choose project(s) to run a focused pass on all of their photos.</p>';

          resultsDiv.innerHTML = '<p class="meta">Tip: search by trade + activity ("steel decking being welded", "CMU wall grouted", "waterproofing at podium") and narrow with project or date.</p>';

          async function toggleBookmark(photoId, button) {
            try {
              const resp = await fetch('/photos/' + photoId + '/bookmark', { method: 'POST' });
              const data = await resp.json();
              if (resp.ok) {
                const isBookmarked = data.bookmarked === 1;
                button.classList.toggle('bookmarked', isBookmarked);
                button.textContent = isBookmarked ? '🔖' : '🔖';
                button.title = isBookmarked ? 'Remove bookmark' : 'Bookmark this photo';

                // Update lightbox item if present
                const lightboxItem = lightboxItems.find(item => item.id === photoId);
                if (lightboxItem) {
                  lightboxItem.bookmarked = isBookmarked ? 1 : 0;
                }

                // Update latestSearchResults if present
                const searchItem = latestSearchResults.find(item => item.id === photoId);
                if (searchItem) {
                  searchItem.bookmarked = isBookmarked ? 1 : 0;
                }

                // Update latestBookmarkedPhotos if present
                const bookmarkedItem = latestBookmarkedPhotos.find(item => item.id === photoId);
                if (bookmarkedItem) {
                  bookmarkedItem.bookmarked = isBookmarked ? 1 : 0;
                }

                return isBookmarked;
              }
            } catch (err) {
              console.error('Error toggling bookmark:', err);
            }
          }

          async function loadBookmarkedPhotos() {
            bookmarkedPhotos.innerHTML = '<p class="meta">Loading bookmarked photos...</p>';
            try {
              const resp = await fetch('/bookmarked');
              const data = await resp.json();
              latestBookmarkedPhotos = data.photos || [];

              bookmarkedPhotos.innerHTML = '';
              if (!latestBookmarkedPhotos.length) {
                bookmarkedPhotos.innerHTML = '<p class="meta">No bookmarked photos yet. Click the bookmark icon on any photo to save it here.</p>';
                return;
              }

              latestBookmarkedPhotos.forEach((r, idx) => {
                const card = document.createElement('div');
                card.className = 'photo-card';
                card.innerHTML = \`
                  <img src="\${r.image_url}" alt="Photo from \${r.project || 'project'}">
                  <div class="meta">
                    <span class="pill">\${r.project || 'Unknown project'}</span><br/>
                    \${r.shot_date || 'Date unknown'}<br/>
                    <em>\${r.description}</em>
                  </div>
                  <div class="actions">
                    <button type="button" class="bookmark-btn bookmarked" data-id="\${r.id}" title="Remove bookmark">🔖</button>
                    <button type="button" data-url="\${r.image_url}">Open full-size</button>
                  </div>
                \`;
                const preview = card.querySelector('img');
                const openBtn = card.querySelector('button[data-url]');
                const bookmarkBtn = card.querySelector('.bookmark-btn');
                const openFull = () => openLightbox(latestBookmarkedPhotos, idx);
                preview.addEventListener('click', openFull);
                openBtn.addEventListener('click', openFull);
                bookmarkBtn.addEventListener('click', async () => {
                  const isBookmarked = await toggleBookmark(r.id, bookmarkBtn);
                  if (isBookmarked === false) {
                    // Remove from view if unbookmarked
                    card.remove();
                    latestBookmarkedPhotos = latestBookmarkedPhotos.filter(item => item.id !== r.id);
                    if (!latestBookmarkedPhotos.length) {
                      bookmarkedPhotos.innerHTML = '<p class="meta">No bookmarked photos yet. Click the bookmark icon on any photo to save it here.</p>';
                    }
                  }
                });
                bookmarkedPhotos.appendChild(card);
              });
            } catch (err) {
              bookmarkedPhotos.innerHTML = '<p class="meta">Error loading bookmarked photos.</p>';
              console.error('Error loading bookmarked photos:', err);
            }
          }

          function setActiveTab(targetId) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
              btn.classList.toggle('active', btn.dataset.tabTarget === targetId);
            });
            document.querySelectorAll('.tab-panel').forEach(panel => {
              panel.classList.toggle('active', panel.id === targetId);
            });
            if (targetId === 'projectsTab' || targetId === 'focusTab') {
              loadProjects();
            }
            if (targetId === 'bookmarkedTab') {
              loadBookmarkedPhotos();
            }
          }

          document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => setActiveTab(btn.dataset.tabTarget));
          });

          function openLightbox(items, index) {
            lightboxItems = items;
            lightboxIndex = index;
            renderLightbox();
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden';
          }

          function closeLightbox() {
            lightbox.classList.remove('active');
            document.body.style.overflow = '';
          }

          function renderLightbox() {
            if (!lightboxItems.length) return;
            const current = lightboxItems[lightboxIndex];
            lightboxImage.src = current.image_url;
            lightboxImage.alt = current.description || 'Photo detail';
            lightboxTitle.textContent = (current.project || 'Project') + ' — ' + (current.shot_date || 'Date unknown');
            lightboxCaption.textContent = current.description || 'No description available.';
            lightboxCounter.textContent = (lightboxIndex + 1) + ' of ' + lightboxItems.length;

            // Update bookmark button state
            const isBookmarked = current.bookmarked === 1;
            lightboxBookmark.classList.toggle('bookmarked', isBookmarked);
            lightboxBookmark.title = isBookmarked ? 'Remove bookmark' : 'Bookmark this photo';
            lightboxBookmark.dataset.photoId = current.id;
          }

          closeLightboxBtn.addEventListener('click', closeLightbox);
          lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox) closeLightbox();
          });
          lightboxBookmark.addEventListener('click', () => {
            const photoId = parseInt(lightboxBookmark.dataset.photoId, 10);
            if (photoId) {
              toggleBookmark(photoId, lightboxBookmark);
            }
          });
          prevLightboxBtn.addEventListener('click', () => {
            if (!lightboxItems.length) return;
            lightboxIndex = (lightboxIndex - 1 + lightboxItems.length) % lightboxItems.length;
            renderLightbox();
          });
          nextLightboxBtn.addEventListener('click', () => {
            if (!lightboxItems.length) return;
            lightboxIndex = (lightboxIndex + 1) % lightboxItems.length;
            renderLightbox();
          });

          function openContextModal(passes, title) {
            contextTitle.textContent = title;
            contextList.innerHTML = '';
            (passes || []).forEach((p) => {
              const item = document.createElement('div');
              item.className = 'context-item';
              item.innerHTML =
                '<strong>' + (p.label || 'Pass') + '</strong>' +
                '<div class="meta">' + (p.description || 'No description available.') + '</div>' +
                '<div class="soft">' + (p.created_at || '') + '</div>';
              contextList.appendChild(item);
            });
            contextModal.classList.add('active');
            document.body.style.overflow = 'hidden';
          }

          function closeContextModal() {
            contextModal.classList.remove('active');
            document.body.style.overflow = '';
          }

          closeContextBtn.addEventListener('click', closeContextModal);
          contextModal.addEventListener('click', (e) => {
            if (e.target === contextModal) closeContextModal();
          });

          let selectedFocusProjects = [];
          let focusSelectedPhotoIds = [];

          function getSelectedFocusProjects() {
            return Array.from(document.querySelectorAll('input[name="focusProject"]:checked')).map(
              (c) => c.value
            );
          }

          async function loadProjects() {
            const resp = await fetch('/projects');
            const data = await resp.json();
            projectsList.innerHTML = '';
            existingProjectSelect.innerHTML = '';
            focusProjectList.innerHTML = '';

            const defaultOpt = document.createElement('option');
            defaultOpt.value = '';
            defaultOpt.textContent = data.projects?.length ? 'Select a project' : 'No projects yet';
            existingProjectSelect.appendChild(defaultOpt);

            const prevSelection = new Set(selectedFocusProjects);

            (data.projects || []).forEach((p) => {
              const opt = document.createElement('option');
              opt.value = p.project;
              opt.textContent = p.project;
              existingProjectSelect.appendChild(opt);

              const focusLabelEl = document.createElement('label');
              focusLabelEl.className = 'checkbox-card';
              focusLabelEl.innerHTML =
                '<input type="checkbox" name="focusProject" value="' + p.project + '" />' +
                '<div>' +
                '<strong>' + p.project + '</strong>' +
                '<div class="soft">' +
                (p.count || 0) + ' photo(s) • ' + (p.pass_count || 0) + ' pass(es)' +
                '</div>' +
                '<div class="soft">' + (p.earliest || '') + ' - ' + (p.latest || '') + '</div>' +
                '</div>';
              focusProjectList.appendChild(focusLabelEl);

              const chip = document.createElement('div');
              chip.className = 'project-chip';
              chip.innerHTML = \`
                <div>
                  <strong>\${p.project}</strong><br/>
                  <span class="soft">\${p.count} photo(s) • \${p.pass_count || 0} pass(es)</span>
                </div>
                <div class="soft">\${p.earliest || ''} - \${p.latest || ''}</div>
              \`;
              chip.addEventListener('click', () => loadProjectPhotos(p.project));
              projectsList.appendChild(chip);
            });

            Array.from(document.querySelectorAll('input[name="focusProject"]')).forEach((input) => {
              input.checked = prevSelection.has(input.value);
            });

            selectedFocusProjects = getSelectedFocusProjects();
            loadFocusPhotosForProjects(selectedFocusProjects);
          }

          async function loadProjectPhotos(project) {
            if (!project) return;
            projectPhotos.innerHTML = '<p class="meta">Loading photos for ' + project + '...</p>';
            const encoded = encodeURIComponent(project);
            const resp = await fetch('/projects/' + encoded + '/photos');
            const data = await resp.json();

            projectPhotos.innerHTML = '';
            (data.photos || []).forEach((r, idx) => {
              const card = document.createElement('div');
              card.className = 'photo-card';
              const isBookmarked = r.bookmarked === 1;
              card.innerHTML =
                '<img src="' + r.image_url + '" alt="Photo from ' + (r.project || 'project') + '">' +
                '<div class="meta">' +
                '<span class="pill">' + (r.project || 'Unknown project') + '</span><br/>' +
                (r.shot_date || 'Date unknown') +
                '</div>' +
                '<div class="actions">' +
                '<div class="stack">' +
                '<button type="button" class="bookmark-btn ' + (isBookmarked ? 'bookmarked' : '') + '" data-bookmark-id="' + r.id + '" title="' + (isBookmarked ? 'Remove bookmark' : 'Bookmark this photo') + '">🔖</button>' +
                '<button type="button" data-context="' + r.id + '">View contexts</button>' +
                '<button type="button" data-url="' + r.image_url + '">Open full-size</button>' +
                '</div>' +
                '<button type="button" data-id="' + r.id + '" class="danger">Delete</button>' +
                '</div>';
              const preview = card.querySelector('img');
              const openBtn = card.querySelector('button[data-url]');
              const contextBtn = card.querySelector('button[data-context]');
              const deleteBtn = card.querySelector('button[data-id]');
              const bookmarkBtn = card.querySelector('.bookmark-btn');
              const openFull = () => openLightbox(data.photos, idx);
              preview.addEventListener('click', openFull);
              openBtn.addEventListener('click', openFull);
              bookmarkBtn.addEventListener('click', () => toggleBookmark(r.id, bookmarkBtn));
              contextBtn.addEventListener('click', () => {
                const title = 'Contexts for ' + (r.project || 'project') + ' (' + (r.shot_date || 'Date unknown') + ')';
                openContextModal(r.passes || [], title);
              });
              deleteBtn.addEventListener('click', async () => {
                deleteBtn.disabled = true;
                deleteBtn.textContent = 'Deleting...';
                const delResp = await fetch('/photos/' + r.id, { method: 'DELETE' });
                if (delResp.ok) {
                  card.remove();
                  loadProjects();
                } else {
                  deleteBtn.disabled = false;
                  deleteBtn.textContent = 'Delete';
                  alert('Failed to delete photo');
                }
              });
              projectPhotos.appendChild(card);
            });

            if (!data.photos?.length) {
              projectPhotos.innerHTML = '<p class="meta">No photos found for that project.</p>';
            }
          }

          async function loadFocusPhotosForProjects(projects) {
            if (!projects.length) {
              focusSelectedPhotoIds = [];
              focusPhotos.innerHTML = '<p class="meta">Choose project(s) to run a focused pass on all photos.</p>';
              return;
            }
            focusPhotos.innerHTML = '<p class="meta">Loading photo counts for ' + projects.join(', ') + '...</p>';

            const projectSummaries = [];
            focusSelectedPhotoIds = [];

            for (const project of projects) {
              try {
                const encoded = encodeURIComponent(project);
                const resp = await fetch('/projects/' + encoded + '/photos');
                const data = await resp.json();
                const photos = data.photos || [];
                const passTotal = photos.reduce((acc, p) => {
                  const focused = (p.passes || []).filter((pass) => !String(pass.id || '').startsWith('base-'));
                  return acc + focused.length;
                }, 0);
                projectSummaries.push({
                  project,
                  photoCount: photos.length,
                  passCount: passTotal,
                });
                photos.forEach((photo) => focusSelectedPhotoIds.push(photo.id));
              } catch (err) {
                console.error('Failed to load project', project, err);
              }
            }

            if (!focusSelectedPhotoIds.length) {
              focusPhotos.innerHTML = '<p class="meta">No photos found for the selected project(s).</p>';
              return;
            }

            focusPhotos.innerHTML = '';

            projectSummaries.forEach((p) => {
              const card = document.createElement('div');
              card.className = 'checkbox-card';
              card.innerHTML =
                '<div>' +
                '<strong>' + p.project + '</strong>' +
                '<div class="soft">' + p.photoCount + ' photo(s) selected</div>' +
                '<div class="soft">' + p.passCount + ' pass(es) already saved</div>' +
                '</div>';
              focusPhotos.appendChild(card);
            });

            const totalPhotos = focusSelectedPhotoIds.length;
            const summary = document.createElement('p');
            summary.className = 'meta';
            summary.textContent = 'All ' + totalPhotos + ' photo(s) across the selected project(s) will be re-indexed.';
            focusPhotos.appendChild(summary);
          }

          focusProjectList.addEventListener('change', () => {
            focusProgress.style.width = '0%';
            focusStatus.textContent = '';
            selectedFocusProjects = getSelectedFocusProjects();
            loadFocusPhotosForProjects(selectedFocusProjects);
          });

          let focusPoller = null;

          async function startFocusJob() {
            const projects = getSelectedFocusProjects();
            if (!projects.length) {
              focusStatus.textContent = 'Pick at least one project first.';
              return;
            }

            if (!focusSelectedPhotoIds.length) {
              focusStatus.textContent = 'No photos found for the selected project(s).';
              return;
            }

            const payload = {
              photoIds: focusSelectedPhotoIds,
              focus: focusPrompt.value.trim(),
              label: focusLabel.value.trim(),
            };

            runFocusBtn.disabled = true;
            runFocusBtn.textContent = 'Running...';
            focusStatus.textContent = 'Starting focused pass...';
            focusProgress.style.width = '0%';

            try {
              const resp = await fetch('/reindex', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
              });

              if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || 'Failed to start focus job');
              }

              const { jobId } = await resp.json();
              focusStatus.textContent = 'Focus job created. Processing...';

              if (focusPoller) clearInterval(focusPoller);
              focusPoller = setInterval(async () => {
                const pollResp = await fetch('/reindex/' + jobId);
                if (!pollResp.ok) return;
                const job = await pollResp.json();
                const pct = Math.round((job.done / job.total) * 100);
                focusProgress.style.width = pct + '%';
                focusStatus.textContent = 'Processing ' + job.done + '/' + job.total + ' photos...';

                if (job.status === 'completed') {
                  clearInterval(focusPoller);
                  focusPoller = null;
                  const summary = job.errors?.length ? 'Some items failed.' : 'All items processed.';
                  const projectLabel = projects.join(', ');
                  focusStatus.textContent = 'Focused pass complete for ' + projectLabel + '. ' + summary;
                  runFocusBtn.disabled = false;
                  runFocusBtn.textContent = 'Run focused pass';
                  loadFocusPhotosForProjects(projects);
                  loadProjects();
                }
              }, 800);
            } catch (err) {
              focusStatus.textContent = err.message || 'Unable to start focused pass.';
              runFocusBtn.disabled = false;
              runFocusBtn.textContent = 'Run focused pass';
            }
          }

          runFocusBtn.addEventListener('click', startFocusJob);

          uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            uploadStatus.textContent = '';
            uploadProgress.style.width = '0%';
            if (indexingTimer) {
              clearInterval(indexingTimer);
              indexingTimer = null;
            }

            const mode = document.querySelector('input[name="projectMode"]:checked').value;
            const project = mode === 'existing'
              ? existingProjectSelect.value
              : document.getElementById('newProject').value.trim();

            if (!project) {
              uploadStatus.textContent = 'Please select or enter a project name.';
              return;
            }

            const photosInput = document.getElementById('photos');
            if (!photosInput.files.length) {
              uploadStatus.textContent = 'Please choose at least one photo.';
              return;
            }

            const fd = new FormData();
            fd.append('project', project);
            const shotDate = document.getElementById('shotDate').value;
            if (shotDate) fd.append('shot_date', shotDate);
            Array.from(photosInput.files).forEach(f => fd.append('photos', f));

            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload');
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Uploading...';

            xhr.upload.addEventListener('progress', (event) => {
              if (event.lengthComputable) {
                const pct = Math.round((event.loaded / event.total) * 100);
                uploadProgress.style.width = pct + '%';
                uploadStatus.textContent = 'Uploading ' + pct + '%';
              }
            });

            xhr.upload.addEventListener('load', () => {
              uploadStatus.textContent = 'Upload complete. Indexing photos...';
              let pct = Math.max(60, parseInt(uploadProgress.style.width, 10) || 0);
              uploadProgress.style.width = pct + '%';
              indexingTimer = setInterval(() => {
                pct = Math.min(95, pct + 1.5);
                uploadProgress.style.width = pct + '%';
              }, 300);
            });

            xhr.onload = () => {
              uploadBtn.disabled = false;
              uploadBtn.textContent = 'Upload & Index';
              if (indexingTimer) {
                clearInterval(indexingTimer);
                indexingTimer = null;
              }
              uploadProgress.style.width = '100%';
              if (xhr.status >= 200 && xhr.status < 300) {
                uploadStatus.textContent = 'Upload complete. Indexing finished for ' + project + '.';
                photosInput.value = '';
                setActiveTab('projectsTab');
                loadProjects();
              } else {
                uploadStatus.textContent = 'Upload failed: ' + xhr.responseText;
              }
            };

            xhr.onerror = () => {
              uploadBtn.disabled = false;
              uploadBtn.textContent = 'Upload & Index';
              if (indexingTimer) {
                clearInterval(indexingTimer);
                indexingTimer = null;
              }
              uploadStatus.textContent = 'Upload failed. Please retry.';
            };

            xhr.send(fd);
          });

          searchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const q = document.getElementById('q').value;
            const project = document.getElementById('projectFilter').value;
            const shotDate = document.getElementById('shotDateFilter').value;

            resultsDiv.innerHTML = '<p class="meta">Searching...</p>';

            const params = new URLSearchParams();
            params.set('q', q);
            if (project) params.set('project', project);
            if (shotDate) params.set('shot_date', shotDate);

            const resp = await fetch('/search?' + params.toString());
            const data = await resp.json();

            latestSearchResults = data.results || [];
            resultsDiv.innerHTML = '';
            latestSearchResults.forEach((r, idx) => {
              const card = document.createElement('div');
              card.className = 'photo-card';
              const focusedCount = (r.passes || []).filter((p) => !String(p.id || '').startsWith('base-')).length;
              const isBookmarked = r.bookmarked === 1;
              card.innerHTML = \`
                <img src="\${r.image_url}" alt="Photo from \${r.project || 'project'}">
                <div class="meta">
                  <span class="pill">\${r.project || 'Unknown project'}</span><br/>
                  \${r.shot_date || 'Date unknown'}<br/>
                  <em>\${r.description}</em>
                  <div class="soft">Matched pass: \${r.best_pass_label || 'Base'}</div>
                  <div class="soft">Passes: \${(r.passes || []).length} total (\${focusedCount} focused)</div>
                </div>
                <div class="actions">
                  <div class="stack">
                    <button type="button" class="bookmark-btn \${isBookmarked ? 'bookmarked' : ''}" data-id="\${r.id}" title="\${isBookmarked ? 'Remove bookmark' : 'Bookmark this photo'}">🔖</button>
                    <span class="score">Score \${r.score.toFixed(3)}</span>
                  </div>
                  <button type="button" data-url="\${r.image_url}">Open full-size</button>
                </div>
              \`;
              const preview = card.querySelector('img');
              const openBtn = card.querySelector('button[data-url]');
              const bookmarkBtn = card.querySelector('.bookmark-btn');
              const openFull = () => openLightbox(latestSearchResults, idx);
              preview.addEventListener('click', openFull);
              openBtn.addEventListener('click', openFull);
              bookmarkBtn.addEventListener('click', () => toggleBookmark(r.id, bookmarkBtn));
              resultsDiv.appendChild(card);
            });

            if (!latestSearchResults.length) {
              resultsDiv.innerHTML = '<p class="meta">No strong matches found for that search.</p>';
            }
          });

          loadProjects();
        </script>
      </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log("Server listening on http://localhost:" + PORT);
});
