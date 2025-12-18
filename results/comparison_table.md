# GraphRAG vs Naive RAG Comparison Results

## Summary
- **GraphRAG Wins**: 15
- **Naive RAG Wins**: 0
- **Ties**: 5
- **GraphRAG Win Rate**: 75.0%

- **GraphRAG Avg Quality**: 0.65
- **Naive RAG Avg Quality**: 0.09

## Performance by Category

### Listing
- GraphRAG: 5/9 (55.6%)

### Counting
- GraphRAG: 5/6 (83.3%)

### Filtering
- GraphRAG: 2/2 (100.0%)

### Aggregation
- GraphRAG: 3/3 (100.0%)

## Detailed Results

| # | Question | Category | GraphRAG Answer | Naive RAG Answer | Ground Truth | Winner |
|---|----------|----------|-----------------|------------------|--------------|--------|
| 1 | List developers with their project counts and univ... | listing | Here are the developers along ... | I don't know. | Darius Villa: Project Count: 4... | GraphRAG ✅ |
| 2 | Give me best developers based on project counts an... | listing | Here are the best developers b... | I don't know. | Victor Cook (Projects: 5; Univ... | GraphRAG ✅ |
| 3 | How many Python developers we have ? | counting | We have 3 Python developers. | I don't know. The context docu... | We have 3 Python developers. | GraphRAG ✅ |
| 4 | Find senior developers with React AND Node.js expe... | filtering | I couldn't find the specific a... | I don't know. | I'm unable to find specific in... | GraphRAG ✅ |
| 5 | How many Python developers available Q2? | counting | There are 3 Python developers ... | I don't know. | There are 3 Python developers ... | GraphRAG ✅ |
| 6 | Number of years of experience for developers | aggregation | The average number of years of... | The context documents do not s... | The average number of years of... | GraphRAG ✅ |
| 7 | Total capacity available for Q4 projects | aggregation | The total capacity available f... | I don't know. | The total capacity available f... | GraphRAG ✅ |
| 8 | Developers from same university | listing | I couldn't find specific infor... | I don't know. | There are no developers from t... | Tie ⚖️ |
| 9 | Give me one python developer | listing | It seems that there are curren... | I don't know. | One Python developer is Mary H... | GraphRAG ✅ |
| 10 | Give me titles of all Rfps | listing | The titles of all RFPs are as ... | The titles of all RFPs are:

1... | Data Analytics Platform Develo... | Tie ⚖️ |
| 11 | count available people with that ptyhon skill for ... | counting | There are 3 available people w... | I don't know. | There are 3 available people w... | GraphRAG ✅ |
| 12 | How many rpfs we have ? | counting | Error: {code: Neo.ClientError.... | There are three RFPs available... | We have a total of 3 RFPs. | Tie ⚖️ |
| 13 | How many Java developers we have ? | counting | We have 1 Java developer. | I don't know. The context does... | We have 1 Java developer. | GraphRAG ✅ |
| 14 | Give me all rpfs titles ? | listing | The RFP titles are as follows:... | The titles of the RFPs are as ... | Data Analytics Platform Develo... | Tie ⚖️ |
| 15 | Number of years of experience for python developer... | aggregation | The average number of years of... | I don't know. | The average number of years of... | GraphRAG ✅ |
| 16 | List all developers in Pacific timezone | filtering | I couldn't find specific infor... | I don't know. | I'm sorry, but I couldn't find... | GraphRAG ✅ |
| 17 | List all skils only name of a skill | listing | The skills available are:

- J... | 1. Java
2. TypeScript
3. Djang... | Javascript; React; Angular; Mi... | Tie ⚖️ |
| 18 | How many people with docker skill we have ? | counting | There are 5 developers with Do... | The context does not specify t... | We have 5 people with Docker s... | GraphRAG ✅ |
| 19 | List me one person with python and docker skill | listing | Here are some developers with ... | I don't know. | One person with Python and Doc... | GraphRAG ✅ |
| 20 | Give me university for Mark Carroll | listing | Mark Carroll attended the Univ... | I don't know. | Mark Carroll attended the Univ... | GraphRAG ✅ |