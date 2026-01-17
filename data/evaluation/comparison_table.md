# GraphRAG vs Naive RAG Comparison Results

## Summary
- **GraphRAG Wins**: 19
- **Naive RAG Wins**: 1
- **Ties**: 0
- **GraphRAG Win Rate**: 95.0%

- **GraphRAG Avg Quality**: 0.72
- **Naive RAG Avg Quality**: 0.09

## Performance by Category

### Listing
- GraphRAG: 8/9 (88.9%)

### Counting
- GraphRAG: 6/6 (100.0%)

### Filtering
- GraphRAG: 2/2 (100.0%)

### Aggregation
- GraphRAG: 3/3 (100.0%)

## Detailed Results

| # | Question | Graph Answer | Naive Answer | Graph Score | Naive Score | Winner |
|---|----------|--------------|--------------|-------------|-------------|--------|
| 1 | List developers with their project count... | Here are some developers along... | I don't know. | 0.85 | 0.00 | GraphRAG |
| 2 | Give me best developers based on project... | Here are the best developers b... | I don't know. | 0.61 | 0.00 | GraphRAG |
| 3 | How many Python developers we have ? | We have 3 Python developers. I... | I don't know. | 0.88 | 0.00 | GraphRAG |
| 4 | Find senior developers with React AND No... | I'm unable to retrieve the spe... | I don't know. | 0.70 | 0.00 | GraphRAG |
| 5 | How many Python developers available Q2? | There are 3 Python developers ... | I don't know. | 0.90 | 0.00 | GraphRAG |
| 6 | Number of years of experience for develo... | The average number of years of... | I don't know. | 0.84 | 0.00 | GraphRAG |
| 7 | Total capacity available for Q4 projects | The total capacity available f... | I don't know. | 0.92 | 0.00 | GraphRAG |
| 8 | Developers from same university | Here are the developers groupe... | I don't know. | 0.70 | 0.00 | GraphRAG |
| 9 | Give me one python developer | One Python developer is **Mary... | I don't know. | 0.86 | 0.00 | GraphRAG |
| 10 | Give me titles of all Rfps | The titles of all RFPs are as ... | 1. Data Analytics Platform Dev... | 0.31 | 0.44 | Naive RAG |
| 11 | count available people with that ptyhon ... | There are 3 available develope... | I don't know. | 0.73 | 0.00 | GraphRAG |
| 12 | How many rpfs we have ? | We have a total of 3 distinct ... | There are a total of 3 RFPs. | 0.82 | 0.32 | GraphRAG |
| 13 | How many Java developers we have ? | We have 4 Java developers avai... | I don't know. | 0.37 | 0.00 | GraphRAG |
| 14 | Give me all rpfs titles ? | The RFP titles are as follows:... | The titles of all RFPs are:  1... | 0.64 | 0.52 | GraphRAG |
| 15 | Number of years of experience for python... | The average number of years of... | I don't know. | 0.88 | 0.00 | GraphRAG |
| 16 | List all developers in Pacific timezone | I'm currently unable to retrie... | I don't know. | 0.73 | 0.13 | GraphRAG |
| 17 | List all skils only name of a skill | Here are the skills available:... | The skills mentioned in the co... | 0.70 | 0.46 | GraphRAG |
| 18 | How many people with docker skill we hav... | We have 5 developers with Dock... | I don't know. | 0.76 | 0.00 | GraphRAG |
| 19 | List me one person with python and docke... | One person with both Python an... | I don't know. | 0.37 | 0.00 | GraphRAG |
| 20 | Give me university for Mark Carroll | Mark Carroll graduated from th... | I don't know. | 0.75 | 0.00 | GraphRAG |