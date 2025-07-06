# FUNDizzle - AI-Powered Grant Matching Platform

A Flask-based web application that uses AI embeddings to match researchers and organizations with relevant funding opportunities. The platform leverages OpenAI's embedding models and PostgreSQL with vector similarity search to provide intelligent grant recommendations.

## Features

- **AI-Powered Grant Matching**: Uses OpenAI embeddings to match users with relevant grants based on their profiles and research interests
- **Multi-Dimensional Embeddings**: Supports both 1536 and 3072-dimensional embeddings for different levels of precision
- **JWT Authentication**: Secure user authentication using Supabase JWT tokens
- **Vector Similarity Search**: PostgreSQL with pgvector extension for efficient similarity matching
- **Grant Management**: Upload, store, and manage grant data with structured metadata
- **User Profiles**: Comprehensive user profile management with research interests and keywords
- **Saved Grants**: Users can save and manage their favorite grants
- **Reaction System**: Track user reactions to grant recommendations

## Architecture

- **Backend**: Flask REST API
- **Database**: PostgreSQL with pgvector extension
- **Authentication**: Supabase JWT
- **AI/ML**: OpenAI Embeddings API
- **Vector Search**: Cosine similarity using pgvector

## API Endpoints

### Authentication Required Endpoints

- `POST /match-grants` - Match user profile with relevant grants
- `POST /update-reaction` - Update user reaction to a grant
- `GET /saved-grants` - Get user's saved grants

### Public Endpoints

- `POST /match-query` - Match a text query with relevant grants

## Setup and Installation

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key
- Supabase account and project

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
PG_DBNAME=your_database_name
PG_USER=your_database_user
PG_PASSWORD=your_database_password
PG_HOST=your_database_host
PG_PORT=5432

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# User Management (for upload_user.py)
PG_EMAIL=your_email
PG_uPASS=your_password
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd inhouse_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
   - Ensure PostgreSQL is running with pgvector extension
   - Create the necessary tables (see Database Schema section)

4. Run the application:
```bash
python app.py
```

## Database Schema

The application uses several key tables:

- `grants` - Grant metadata and information
- `grant_raw_json` - Raw grant data storage
- `grant_embeddings_1536` - 1536-dimensional grant embeddings
- `grant_embeddings_3072` - 3072-dimensional grant embeddings
- `user_details` - Basic user information
- `user_profiles` - Extended user profile data
- `user_embeddings_1536` - 1536-dimensional user embeddings
- `user_embeddings_3072` - 3072-dimensional user embeddings
- `saved_grants` - User's saved grants
- `grant_reactions` - User reactions to grants

## Usage

### Starting the Application

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`

### Uploading Grants

Use the `upload_grants.py` script to populate the database with grant data:

```bash
python upload_grants.py
```

### Setting Up Users

Use the `upload_user.py` script to create user profiles:

```bash
python upload_user.py
```

## API Usage Examples

### Match User with Grants

```bash
curl -X POST http://localhost:5000/match-grants \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dimension": 1536,
    "top_k": 10,
    "offset": 0
  }'
```

### Match Query with Grants

```bash
curl -X POST http://localhost:5000/match-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neuroscience research funding",
    "dimension": 1536,
    "top_k": 10,
    "offset": 0
  }'
```

## Development

### Project Structure

```
inhouse_app/
├── app.py                 # Main Flask application
├── upload_grants.py       # Grant data upload script
├── upload_user.py         # User profile setup script
├── backups/               # Backup and example files
│   ├── main_single-user-embed.py
│   └── requests/          # Example API requests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Key Components

- **Embedding Generation**: Uses OpenAI's text-embedding-3-small (1536d) and text-embedding-3-large (3072d)
- **Similarity Scoring**: Cosine similarity scaled to 5-99 range
- **Rate Limiting**: Built-in delays to respect OpenAI API limits
- **Error Handling**: Comprehensive error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For support and questions, please contact [your contact information]. 