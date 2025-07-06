import json
import os
import psycopg2
from supabase import create_client, Client
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import jwt
from jwt import InvalidTokenError
import sys

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
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY grant_uid ORDER BY similarity DESC) AS rank
                FROM scored_pairs
            ),
            top_matches AS (
                SELECT * FROM best_per_grant WHERE rank = 1
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

@app.route("/update-reaction", methods=["POST"])
# @jwt_required  # assumes you have this decorator or use inline JWT logic
def update_reaction():
    try:
        user_info = verify_jwt(request.headers.get("Authorization", "").split(" ")[1])
        if not user_info:
            return jsonify({"error": "Invalid or expired token"}), 403

        user_id = user_info["sub"]
        body = request.get_json()

        grant_uid = body.get("grant_uid")
        reaction_type = body.get("reaction_type", "neutral")
        is_saved = body.get("is_saved", False)
        reason = body.get("reason")

        if not grant_uid:
            return jsonify({"error": "grant_uid is required"}), 400

        # Validate enum manually
        valid_reactions = ["like", "dislike", "neutral"]
        if reaction_type not in valid_reactions:
            return jsonify({"error": "Invalid reaction_type"}), 400

        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

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

        cursor.execute(upsert_query, (user_id, grant_uid, reaction_type, is_saved, reason))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "reaction_type": reaction_type, "is_saved": is_saved})

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


# For local testing
if __name__ == "__main__":
    app.run(debug=True)
