import json
import os
import psycopg2
from supabase import create_client, Client
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import jwt
from jwt import InvalidTokenError
import sys
from datetime import datetime
import re

load_dotenv()

# PostgreSQL DB config
db_config = {
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT", "5432"),
    "sslmode": "require"
}

def scale_cosine_similarity_to_score(sim):
    if sim is None:
        return 0  # or some fallback value like 5
    return round(5 + sim * (99 - 5))


def verify_jwt(token):
    try:
        SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
        decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        return decoded  # includes user ID, email, etc.
    except InvalidTokenError as e:
        print("JWT error:", e)
        return None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)

@app.route("/match-grants", methods=["POST"])
def match_grants():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]
        body = request.get_json()
        # user_id = body.get("user_id")
        dimension = int(body.get("dimension", 1536))
        top_k = int(body.get("top_k", 10))
        offset = int(body.get("offset", 0))

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        if dimension not in [1536, 3072]:
            return jsonify({"error": "Invalid embedding dimension"}), 400

        # Dynamic table names
        user_embedding_table = f"user_embeddings_{dimension}"
        grant_embedding_table = f"grant_embeddings_{dimension}"

        # Connect to DB
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        match_query = f"""
            WITH scored_pairs AS (
                SELECT 
                    g.grant_uid,
                    g.title,
                    g.agency,
                    g.funding_amount,
                    g.end_date,
                    g.research_areas,
                    1 - (ge.embedding <=> ue.embedding) AS similarity
                FROM {grant_embedding_table} ge
                JOIN grants g ON g.grant_uid = ge.grant_uid
                JOIN {user_embedding_table} ue ON ue.user_id = %s
            ),
            best_per_grant AS (
                SELECT 
                    grant_uid,
                    title,
                    agency,
                    funding_amount,
                    end_date,
                    research_areas,
                    MAX(similarity) AS similarity
                FROM scored_pairs
                GROUP BY grant_uid, title, agency, funding_amount, end_date, research_areas
            ),
            top_matches AS (
                SELECT * FROM best_per_grant
                ORDER BY similarity DESC
                LIMIT %s OFFSET %s
            )
            SELECT * FROM top_matches;
        """

        cursor.execute(match_query, (user_id, top_k, offset))
        matches = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in matches]
        print(results)
        sys.stdout.flush()

        results = []
        for row in matches:
            row_dict = dict(zip(columns, row))
            sim = row_dict.get("similarity", 0.0) or 0.0
            row_dict["match_score"] = scale_cosine_similarity_to_score(sim)
            results.append(row_dict)

        cursor.close()
        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/match-query", methods=["POST"])
def match_query():
    try:
        body = request.get_json()
        query = body.get("query", "")
        dimension = int(body.get("dimension", 1536))
        top_k = int(body.get("top_k", 10))
        offset = int(body.get("offset", 0))

        if not query:
            return jsonify({"error": "Query text is required"}), 400
        if dimension not in [1536, 3072]:
            return jsonify({"error": "Invalid embedding dimension"}), 400

        # Get embedding from OpenAI
        # Choose embedding model based on dimension
        if dimension == 1536:
            model = "text-embedding-3-small"
        elif dimension == 3072:
            model = "text-embedding-3-large"
        else:
            return jsonify({"error": "Unsupported embedding dimension"}), 400

        # Get embedding from OpenAI
        response = openai.embeddings.create(
            input=[query],
            model=model
        )
        query_embedding = response.data[0].embedding

        query_vector = response.data[0].embedding

        grant_embedding_table = f"grant_embeddings_{dimension}"

        # Connect to DB
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query_vector_str = json.dumps(query_vector)

        match_query = f"""
            WITH scored_chunks AS (
                SELECT 
                    g.grant_uid,
                    g.title,
                    g.agency,
                    g.funding_amount,
                    g.end_date,
                    g.research_areas,
                    1 - (ge.embedding <=> %s::vector) AS similarity
                FROM {grant_embedding_table} ge
                JOIN grants g ON g.grant_uid = ge.grant_uid
            ),
            best_per_grant AS (
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY grant_uid ORDER BY similarity DESC) AS rank
                    FROM scored_chunks
                ) ranked WHERE rank = 1
            )
            SELECT * FROM best_per_grant
            ORDER BY similarity DESC
            LIMIT %s OFFSET %s;
        """

        cursor.execute(match_query, (query_vector_str, top_k, offset))
        matches = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = []

        for row in matches:
            row_dict = dict(zip(columns, row))
            sim = row_dict.get("similarity", 0.0) or 0.0
            row_dict["match_score"] = scale_cosine_similarity_to_score(sim)
            results.append(row_dict)

        cursor.close()
        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/grants/<grant_id>", methods=["GET"])
def get_grant_details(grant_id):
    try:
        # Optional JWT check
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.replace("Bearer ", "") if "Bearer " in auth_header else None
        user_id = None

        if token:
            user_info = verify_jwt(token)
            if user_info:
                user_id = user_info.get("sub")

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Fetch grant details
        cursor.execute("SELECT raw_payload FROM grant_raw_json WHERE grant_uid = %s", (grant_id,))
        grant_row = cursor.fetchone()

        if not grant_row:
            return jsonify({"error": "Grant not found"}), 404

        grant_data = {
            "raw_json": grant_row[0]
        }

        # If user is authenticated, fetch user-specific reaction
        if user_id:
            cursor.execute(
                """
                SELECT reaction_type, is_saved, reason, updated_at
                FROM user_reaction
                WHERE user_id = %s AND grant_uid = %s
                """,
                (user_id, grant_id)
            )
            user_row = cursor.fetchone()
            if user_row:
                grant_data["user_reaction"] = {
                    "reaction_type": user_row[0],
                    "is_saved": user_row[1],
                    "reason": user_row[2],
                    "updated_at": user_row[3].isoformat() if user_row[3] else None
                }
            else:
                grant_data["user_reaction"] = None

        cursor.close()
        conn.close()
        return jsonify(grant_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update-reaction", methods=["POST"])
def update_reaction():
    try:
        # --- Step 1: Auth ---
        user_info = verify_jwt(request.headers.get("Authorization", "").split(" ")[1])
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403
        user_id = user_info["sub"]

        # --- Step 2: Input Parsing ---
        body = request.get_json()
        grant_uid = body.get("grant_uid")
        reaction_type = body.get("reaction_type")  # optional
        is_saved = body.get("is_saved")            # optional
        reason = body.get("reason")                # optional

        if not grant_uid:
            return jsonify({"error": "grant_uid is required"}), 400

        valid_reactions = ["like", "dislike", "neutral"]
        if reaction_type and reaction_type not in valid_reactions:
            return jsonify({"error": "Invalid reaction_type. Must be like, dislike, or neutral"}), 400

        # --- Step 3: Connect & Fetch Existing Row ---
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        select_query = """
            SELECT reaction_type, is_saved, reason
            FROM user_reaction
            WHERE user_id = %s AND grant_uid = %s;
        """
        cursor.execute(select_query, (user_id, grant_uid))
        existing = cursor.fetchone()

        # Determine values to use (existing or new)
        if existing:
            existing_reaction_type, existing_is_saved, existing_reason = existing
            final_reaction_type = reaction_type if reaction_type is not None else existing_reaction_type
            final_is_saved = is_saved if is_saved is not None else existing_is_saved
            final_reason = reason if reason is not None else existing_reason
        else:
            # New entry — provide defaults
            final_reaction_type = reaction_type if reaction_type is not None else "neutral"
            final_is_saved = is_saved if is_saved is not None else False
            final_reason = reason if reason is not None else None

        # --- Step 4: Upsert ---
        upsert_query = """
            INSERT INTO user_reaction (user_id, grant_uid, reaction_type, is_saved, reason, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, grant_uid)
            DO UPDATE SET
                reaction_type = EXCLUDED.reaction_type,
                is_saved = EXCLUDED.is_saved,
                reason = EXCLUDED.reason,
                updated_at = NOW();
        """

        cursor.execute(upsert_query, (user_id, grant_uid, final_reaction_type, final_is_saved, final_reason))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "reaction_type": final_reaction_type,
            "is_saved": final_is_saved,
            "reason": final_reason
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/user-reactions", methods=["POST"])
def get_user_reactions_for_grants():
    try:
        # 1. Verify token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.split(" ")[1]
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Parse input
        body = request.get_json()
        grant_uids = body.get("grant_uids", [])

        if not grant_uids:
            return jsonify({"error": "grant_uids must be provided as a non-empty list"}), 400

        # 3. Query DB
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query = """
            SELECT grant_uid, reaction_type, is_saved, reason, updated_at
            FROM user_reaction
            WHERE user_id = %s AND grant_uid = ANY(%s::uuid[])
        """

        cursor.execute(query, (user_id, grant_uids))
        rows = cursor.fetchall()

        result = {}
        for row in rows:
            grant_uid, reaction_type, is_saved, reason, updated_at = row
            result[str(grant_uid)] = {
                "reaction_type": reaction_type,
                "is_saved": is_saved,
                "reason": reason,
                "updated_at": updated_at.isoformat() if updated_at else None
            }

        cursor.close()
        conn.close()
        print(jsonify(result))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/saved-grants", methods=["GET"])
def get_saved_grants():
    try:
        # 1. Verify JWT
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Connect to DB
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # 3. Query saved grants
        query = """
            SELECT g.grant_uid, g.title, g.agency, g.funding_amount, g.end_date, g.research_areas
            FROM user_reaction ur
            JOIN grants g ON ur.grant_uid = g.grant_uid
            WHERE ur.user_id = %s AND ur.is_saved = true
            ORDER BY ur.updated_at DESC;
        """
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/liked-grants", methods=["GET"])
def get_liked_grants():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query = """
            SELECT g.grant_uid, g.title, g.agency, g.funding_amount, g.end_date, g.research_areas
            FROM user_reaction ur
            JOIN grants g ON ur.grant_uid = g.grant_uid
            WHERE ur.user_id = %s AND ur.reaction_type = 'like'
            ORDER BY ur.updated_at DESC;
        """
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/disliked-grants", methods=["GET"])
def get_disliked_grants():
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        query = """
            SELECT g.grant_uid, g.title, g.agency, g.funding_amount, g.end_date, g.research_areas
            FROM user_reaction ur
            JOIN grants g ON ur.grant_uid = g.grant_uid
            WHERE ur.user_id = %s AND ur.reaction_type = 'dislike'
            ORDER BY ur.updated_at DESC;
        """
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        conn.close()
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/total-grants", methods=["GET"])
def get_total_grants():
    try:
        # 1. Verify JWT
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Connect to DB
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM grant_raw_json;")
        count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return jsonify({"total_grants": count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profile-complete", methods=["GET"])
def get_user_profile():
    try:
        # 1. Verify JWT
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Connect to DB and check profile presence
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # SQL query to count presence in both profile tables
        profile_check_query = """
            SELECT 
                CASE 
                    WHEN COUNT(*) = 2 THEN 100  -- Present in both tables
                    WHEN COUNT(*) = 1 THEN 50   -- Present in one table
                    ELSE 0                       -- Present in none
                END as profile_score
            FROM (
                SELECT user_id FROM user_orcid_profile WHERE user_id = %s
                UNION ALL
                SELECT user_id FROM user_linkedin_profile WHERE user_id = %s
            ) profile_check;
        """
        
        cursor.execute(profile_check_query, (user_id, user_id))
        result = cursor.fetchone()
        profile_score = result[0] if result else 0

        cursor.close()
        conn.close()

        return jsonify({"profile_score": profile_score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-user-details", methods=["GET"])
def get_user_details():
    try:
        # 1. Verify JWT
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Connect to DB and fetch user details
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # SQL query to get user details
        query = """
            SELECT fname, lname, date_of_birth, gender
            FROM user_details
            WHERE user_id = %s;
        """
        
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        # Return user details or empty fields if not found
        if result:
            fname, lname, date_of_birth, gender = result
            # Format the date to ensure it's in yyyy-MM-dd format
            formatted_date = format_date_for_db(str(date_of_birth)) if date_of_birth else ""
            return jsonify({
                "fname": fname or "",
                "lname": lname or "",
                "date_of_birth": formatted_date,
                "gender": gender or ""
            })
        else:
            return jsonify({
                "fname": "",
                "lname": "",
                "date_of_birth": "",
                "gender": ""
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_embedding(text, dim):
    model = "text-embedding-3-large" if dim == 3072 else "text-embedding-3-small"
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def chunk_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def generate_summary(data):
    parts = []
    parts.append(f"Location: {data.get('location', '')}")
    parts.append(f"Headline: {data.get('headline', '')}")
    parts.append(f"Summary: {data.get('summary', '')}")

    for exp in data.get("experience", []):
        parts.append(f"Experience: {exp.get('title', '')} at {exp.get('company', '')} - {exp.get('description', '')}")

    for edu in data.get("education", []):
        parts.append(f"Education: {edu.get('degree', '')} at {edu.get('institution', '')} - {edu.get('description', '')}")

    for pub in data.get("publications", []):
        parts.append(f"Publication: {pub.get('title', '')} in {pub.get('journal', '')} - {pub.get('subtitle', '')}")

    parts.append("Skills: " + ", ".join(data.get("skills", [])))
    parts.append("Keywords: " + ", ".join(data.get("keywords", [])))

    for lang in data.get("languages", []):
        parts.append(f"Language: {lang.get('language', '')} ({lang.get('proficiency', '')})")

    return "\n".join([line.strip() for line in parts if line.strip()])

@app.route("/save-orcid-profile", methods=["POST"])
def save_orcid_profile():
    try:
        # Step 1: Verify token and get user_id
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403
        user_id = user_info["sub"]

        # Step 2: Get profile data
        profile_data = request.get_json()
        raw_json_str = json.dumps(profile_data)

        # Step 3: Generate summary
        summary = generate_summary(profile_data)

        # Step 4: Connect to DB and update user_orcid_profile (delete old, insert new)
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Delete existing profile data for this user
        delete_query = "DELETE FROM user_orcid_profile WHERE user_id = %s"
        cursor.execute(delete_query, (user_id,))

        # Insert new profile data
        insert_query = """
            INSERT INTO user_orcid_profile (user_id, raw_json, summary, updated_at)
            VALUES (%s, %s, %s, NOW())
        """
        cursor.execute(insert_query, (user_id, raw_json_str, summary))
        conn.commit()
        cursor.close()
        conn.close()

        # Step 5: Generate embeddings and store them with chunking
        try:
            # Chunk the summary text
            chunks = chunk_text(summary, max_length=1000)
            
            # Store embeddings in both tables (delete old, insert new)
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Delete existing embeddings for both dimensions
            delete_1536_query = "DELETE FROM user_embeddings_1536 WHERE user_id = %s AND source = 'orcid'"
            cursor.execute(delete_1536_query, (user_id,))
            
            delete_3072_query = "DELETE FROM user_embeddings_3072 WHERE user_id = %s AND source = 'orcid'"
            cursor.execute(delete_3072_query, (user_id,))
            
            # Process each chunk and store embeddings
            for chunk_number, chunk_content in enumerate(chunks):
                chunk_size = len(chunk_content)
                
                # Generate embeddings for this chunk
                embedding_1536 = get_embedding(chunk_content, 1536)
                embedding_3072 = get_embedding(chunk_content, 3072)
                
                # Insert 1536 dimension embedding
                insert_1536_query = """
                    INSERT INTO user_embeddings_1536 (user_id, chunk_size, chunk_number, content, embedding, source, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, 'orcid', NOW())
                """
                cursor.execute(insert_1536_query, (user_id, chunk_size, chunk_number, chunk_content, json.dumps(embedding_1536)))
                
                # Insert 3072 dimension embedding
                insert_3072_query = """
                    INSERT INTO user_embeddings_3072 (user_id, chunk_size, chunk_number, content, embedding, source, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, 'orcid', NOW())
                """
                cursor.execute(insert_3072_query, (user_id, chunk_size, chunk_number, chunk_content, json.dumps(embedding_3072)))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Continue without embeddings if there's an error

        return jsonify({"status": "success", "message": "Profile saved and embedded."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save-linkedin-profile", methods=["POST"])
def save_linkedin_profile():
    try:
        # Step 1: Verify token and get user_id
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth_header.replace("Bearer ", "")
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403
        user_id = user_info["sub"]

        # Step 2: Get profile data
        profile_data = request.get_json()
        raw_json_str = json.dumps(profile_data)

        # Step 3: Convert entire JSON to text for summary
        summary = json.dumps(profile_data, ensure_ascii=False, indent=2)

        # Step 4: Connect to DB and update user_linkedin_profile (delete old, insert new)
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Delete existing profile data for this user
        delete_query = "DELETE FROM user_linkedin_profile WHERE user_id = %s"
        cursor.execute(delete_query, (user_id,))

        # Insert new profile data
        insert_query = """
            INSERT INTO user_linkedin_profile (user_id, raw_json, summary, updated_at)
            VALUES (%s, %s, %s, NOW())
        """
        cursor.execute(insert_query, (user_id, raw_json_str, summary))
        conn.commit()
        cursor.close()
        conn.close()

        # Step 5: Generate embeddings and store them with chunking
        try:
            # Chunk the summary text
            chunks = chunk_text(summary, max_length=1000)
            
            # Store embeddings in both tables (delete old, insert new)
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Delete existing embeddings for both dimensions
            delete_1536_query = "DELETE FROM user_embeddings_1536 WHERE user_id = %s AND source = 'linkedin'"
            cursor.execute(delete_1536_query, (user_id,))
            
            delete_3072_query = "DELETE FROM user_embeddings_3072 WHERE user_id = %s AND source = 'linkedin'"
            cursor.execute(delete_3072_query, (user_id,))
            
            # Process each chunk and store embeddings
            for chunk_number, chunk_content in enumerate(chunks):
                chunk_size = len(chunk_content)
                
                # Generate embeddings for this chunk
                embedding_1536 = get_embedding(chunk_content, 1536)
                embedding_3072 = get_embedding(chunk_content, 3072)
                
                # Insert 1536 dimension embedding
                insert_1536_query = """
                    INSERT INTO user_embeddings_1536 (user_id, chunk_size, chunk_number, content, embedding, source, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, 'linkedin', NOW())
                """
                cursor.execute(insert_1536_query, (user_id, chunk_size, chunk_number, chunk_content, json.dumps(embedding_1536)))
                
                # Insert 3072 dimension embedding
                insert_3072_query = """
                    INSERT INTO user_embeddings_3072 (user_id, chunk_size, chunk_number, content, embedding, source, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, 'linkedin', NOW())
                """
                cursor.execute(insert_3072_query, (user_id, chunk_size, chunk_number, chunk_content, json.dumps(embedding_3072)))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Continue without embeddings if there's an error

        return jsonify({"status": "success", "message": "Profile saved and embedded."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def format_date_for_db(date_str):
    """
    Convert various date formats to yyyy-MM-dd format for database storage.
    Handles formats like:
    - yyyy-MM-dd
    - MM/dd/yyyy
    - MM-dd-yyyy
    - "Wed, 09 Jul 2025 00:00:00 GMT" (JavaScript Date.toUTCString())
    """
    if not date_str:
        return ""
    
    try:
        # Try yyyy-MM-dd format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        try:
            # Try MM/dd/yyyy format
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            try:
                # Try MM-dd-yyyy format
                date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    # Try JavaScript Date.toUTCString() format: "Wed, 09 Jul 2025 00:00:00 GMT"
                    # Remove timezone and parse the date part
                    date_part = date_str.split(', ')[1].split(' ')[0]  # Get "09 Jul 2025"
                    date_obj = datetime.strptime(date_part, '%d %b %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except (ValueError, IndexError):
                    # If all parsing fails, return empty string
                    return ""

@app.route("/save-user-details", methods=["POST"])
def save_user_details():
    try:
        # 1. Verify token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.split(" ")[1]
        user_info = verify_jwt(token)
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]

        # 2. Parse input
        body = request.get_json()
        
        # Validate required fields
        if not body.get("fname") or not body.get("lname"):
            return jsonify({"error": "fname and lname are required"}), 400

        # Format date_of_birth for database storage
        formatted_date = format_date_for_db(body.get("date_of_birth", ""))

        # 3. Connect to DB and upsert user details
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Upsert query to handle both insert and update
        query = """
            INSERT INTO user_details (user_id, fname, lname, date_of_birth, gender, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id)
            DO UPDATE SET
                fname = EXCLUDED.fname,
                lname = EXCLUDED.lname,
                date_of_birth = EXCLUDED.date_of_birth,
                gender = EXCLUDED.gender,
                updated_at = NOW()
        """

        cursor.execute(query, (
            user_id, 
            body.get("fname", ""), 
            body.get("lname", ""), 
            formatted_date, 
            body.get("gender", "")
        ))
        
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success", 
            "message": "User details saved successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# For local testing
if __name__ == "__main__":
    app.run(debug=True)
