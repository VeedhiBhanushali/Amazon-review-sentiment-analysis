# Amazon Reviews Sentiment Analysis

## Project Overview
This project analyzes sentiment in Amazon fine food reviews using advanced Natural Language Processing (NLP) techniques. The analysis provides insights into customer satisfaction, product perception, and identifies patterns in consumer feedback that can drive business decisions.

## Dataset
- **Source**: Amazon Fine Food Reviews dataset
- **Size**: Analyzed 568,454 food reviews from Amazon
- **Features**: Product information, user data, ratings (1-5 stars), review text, and helpfulness votes
- **Time Period**: Reviews spanning multiple years, providing a comprehensive view of consumer sentiment trends

## Methodology
The project employs a dual-model approach to sentiment analysis:

### Models Used
1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
   - Rule-based sentiment analysis tool specifically attuned to social media
   - Provides sentiment scores across four dimensions: negative, neutral, positive, and compound
   - Particularly effective for short, informal text

2. **RoBERTa (Roberta-base-sentiment)**
   - Pre-trained transformer model from cardiffnlp
   - Accounts for both word meaning and contextual relationships
   - More sophisticated understanding of complex language patterns
   - Fine-tuned for sentiment classification tasks

3. **DistilBERT**
   - Lightweight version of BERT used for additional validation
   - Provides efficient sentiment classification with reduced computational requirements

## Key Findings

### Sentiment Distribution
- Strong correlation between star ratings and sentiment scores across models
- 5-star reviews consistently showed high positive sentiment scores
- 1-star reviews demonstrated strong negative sentiment patterns
- Identified nuanced sentiment in 3-star reviews that often contained mixed opinions

### Model Performance
- RoBERTa demonstrated superior performance in capturing subtle sentiment nuances
- VADER provided efficient analysis with good accuracy for straightforward sentiment
- Combined approach leveraged strengths of both models for comprehensive analysis

### Sentiment-Rating Alignment
- Identified cases where sentiment scores diverged from star ratings, revealing:
  - Reviews with positive language but low ratings (often due to specific product issues)
  - Reviews with critical language but high ratings (often acknowledging good aspects despite flaws)
  - These discrepancies provide valuable insights beyond simple star ratings

## Actionable Recommendations

### For Product Development
1. **Focus on frequently mentioned negative aspects** in otherwise positive reviews to address specific product weaknesses
2. **Identify product strengths** from sentiment analysis to emphasize in marketing materials
3. **Track sentiment trends over time** to measure the impact of product changes

### For Marketing
1. **Leverage positive sentiment themes** in marketing campaigns
2. **Address common concerns** proactively in product descriptions
3. **Target specific customer segments** based on sentiment patterns in different product categories

### For Customer Service
1. **Prioritize response to reviews** with mismatched sentiment and ratings
2. **Develop response templates** addressing common sentiment themes
3. **Monitor sentiment shifts** to identify emerging issues quickly

## Future Work
- Expand analysis to include aspect-based sentiment analysis
- Incorporate additional product categories beyond food items
- Develop a real-time sentiment monitoring dashboard
- Explore sentiment differences across product categories and price points

## Technologies Used
- Python
- NLTK
- Transformers (Hugging Face)
- Pandas, NumPy
- Matplotlib, Seaborn

## Conclusion
This sentiment analysis project demonstrates the value of applying advanced NLP techniques to customer reviews. By combining traditional lexicon-based methods (VADER) with state-of-the-art transformer models (RoBERTa), we achieved a nuanced understanding of customer sentiment that goes beyond simple star ratings. The insights generated can drive strategic decisions across product development, marketing, and customer service functions.
