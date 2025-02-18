# Movie Recommendation System

A content-based movie recommendation system that suggests movies based on user text input, including ratings and genre information.

## Dataset

This project uses a subset of TMDB Movies Dataset from Kaggle.


## Features

- Content-based recommendation using TF-IDF and cosine similarity
- Includes movie ratings, genres, and release years
- Genre-aware recommendations with weighted matching
- Recency boost for newer movies
- Enhanced query processing with genre synonyms

## Setup

### Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Pragyan-dev/lumaa-spring-2025-ai-ml.git
cd lumaa-spring-2025-ai-ml
```

2. Create and activate a virtual environment(Optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the recommender with your movie preferences:
```bash
python recommender.py
```

Then enter your preferences when prompted, for example:
- "I like action movies set in space"
- "Show me sci-fi adventures with aliens"
- "Looking for space comedies"


## Project Structure

```
movie-recommender/
│
├── recommender.py      # Main recommendation system
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── movies.csv         # Dataset 

## Requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
```

## Expected Salary

 $20–30/hr

## How It Works

1. **Text Processing**:
   - Converts user input and movie descriptions to TF-IDF vectors
   - Enhances queries with genre synonyms
   - Weights important genres and keywords

2. **Similarity Calculation**:
   - Uses cosine similarity between query and movie vectors
   - Applies recency boost for newer movies
   - Considers genre matching and popularity

3. **Recommendation Generation**:
   - Ranks movies by adjusted similarity scores
   - Returns top matches with detailed information
   - Includes ratings, votes, and genres

