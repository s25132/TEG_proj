# GraphRAG vs Naive RAG Comparison Results

## Summary
- **GraphRAG Wins**: 12
- **Naive RAG Wins**: 2
- **Ties**: 6
- **GraphRAG Win Rate**: 60.0%

- **GraphRAG Avg Quality**: 0.16
- **Naive RAG Avg Quality**: 0.05

## Performance by Category

### Listing
- GraphRAG: 6/9 (66.7%)

### Counting
- GraphRAG: 3/6 (50.0%)

### Filtering
- GraphRAG: 0/2 (0.0%)

### Aggregation
- GraphRAG: 3/3 (100.0%)

## Detailed Results

| # | Question | Graph Answer | Naive Answer | Graph Score | Naive Score | Winner |
|---|----------|--------------|--------------|-------------|-------------|--------|
| 1 | List developers with their project count... | Here are the developers along ... | I don't know. | 0.21 | 0.00 | GraphRAG ✅ |
| 2 | Give me best developers based on project... | Here are the best developers b... | I don't know. | 0.22 | 0.00 | GraphRAG ✅ |
| 3 | How many Python developers we have ? | We have 3 Python developers. I... | I don't know. The provided con... | 0.23 | 0.25 | Naive RAG ✅ |
| 4 | Find senior developers with React AND No... | I couldn't find specific infor... | I don't know. | 0.00 | 0.00 | Tie ⚖️ |
| 5 | How many Python developers available Q2? | There are 3 Python developers ... | I don't know. | 0.25 | 0.00 | GraphRAG ✅ |
| 6 | Number of years of experience for develo... | The average number of years of... | The context documents do not s... | 0.22 | 0.00 | GraphRAG ✅ |
| 7 | Total capacity available for Q4 projects | The total capacity available f... | I don't know. | 0.21 | 0.00 | GraphRAG ✅ |
| 8 | Developers from same university | I couldn't retrieve the specif... | I don't know. | 0.00 | 0.00 | Tie ⚖️ |
| 9 | Give me one python developer | One Python developer is **Mary... | I don't know. | 0.16 | 0.00 | GraphRAG ✅ |
| 10 | Give me titles of all Rfps | The titles of all RFPs are as ... | The titles of all RFPs are:  1... | 0.15 | 0.15 | Tie ⚖️ |
| 11 | count available people with that ptyhon ... | There are 3 available develope... | I don't know. | 0.12 | 0.00 | GraphRAG ✅ |
| 12 | How many rpfs we have ? | There are 3 RFPs available. If... | There are three RFPs available... | 0.15 | 0.15 | Tie ⚖️ |
| 13 | How many Java developers we have ? | We have 1 Java developer. If y... | I don't know. The provided con... | 0.23 | 0.25 | Naive RAG ✅ |
| 14 | Give me all rpfs titles ? | The titles of all RFPs are as ... | The titles of the RFPs are:  1... | 0.16 | 0.15 | Tie ⚖️ |
| 15 | Number of years of experience for python... | The average number of years of... | I don't know. | 0.22 | 0.00 | GraphRAG ✅ |
| 16 | List all developers in Pacific timezone | I couldn't retrieve the specif... | I don't know. | 0.00 | 0.00 | Tie ⚖️ |
| 17 | List all skils only name of a skill | The available skills are as fo... | 1. Java 2. TypeScript 3. Djang... | 0.15 | 0.04 | GraphRAG ✅ |
| 18 | How many people with docker skill we hav... | We have 2 developers with Dock... | I don't know. | 0.21 | 0.00 | GraphRAG ✅ |
| 19 | List me one person with python and docke... | One person with both Python an... | I don't know. | 0.21 | 0.00 | GraphRAG ✅ |
| 20 | Give me university for Mark Carroll | Mark Carroll graduated from th... | I don't know. | 0.19 | 0.00 | GraphRAG ✅ |