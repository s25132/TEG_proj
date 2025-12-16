## Tool graph_qa

Tool umożliwia zadawanie pytań w języku naturalnym do grafowej bazy danych Neo4j
z wykorzystaniem `GraphCypherQAChain`.

### Jak działa
1. Przyjmuje pytanie w języku naturalnym
2. Generuje zapytanie Cypher przy pomocy LLM
3. Wykonuje zapytanie na bazie Neo4j
4. Zwraca:
   - odpowiedź w języku naturalnym
   - wygenerowane zapytanie Cypher
   - kontekst danych użyty do odpowiedzi

## Tool match_devs_to_rfp_scored

Moduł automatycznie dopasowuje developerów do zapytania ofertowego (RFP) na podstawie
wymaganych umiejętności oraz punktowej oceny kandydata.

### Jak działa
1. Odczytuje wymagane umiejętności z grafu Neo4j (`Rfp → NEEDS → Skill`)
2. Dla każdej umiejętności wyszukuje pasujących developerów (`Person → HAS_SKILL → Skill`)
3. Oblicza score developera na podstawie:
   - lat doświadczenia
   - liczby zrealizowanych projektów
   - rankingu uczelni
4. Sortuje kandydatów według score i wybiera najlepszych
5. Zapobiega przypisaniu jednego developera do wielu umiejętności
6. Zwraca przypisania oraz listę nieobsadzonych wymagań


Pytanie: Match devs for API Integration Platform Development  

Odp: Here are the assignments for the "API Integration Platform Development" project:  

Assigned Developers:  

Angular: Jessica Erickson  

Django: Lauren Flowers  

Docker: Darius Villa  

Typescript: Mark Carroll  

Unfilled Skills:  

Angular: Required 3, Filled 1  

Django: Required 2, Filled 1  

Java: Required 2, Filled 0  

Typescript: Required 3, Filled 1  

It looks like there are still some skills that need additional developers. If you need further assistance or adjustments, let me know!  

## Tool rank_best_devs_university

Tool służy do globalnego rankingu developerów na podstawie
doświadczenia zawodowego, liczby zrealizowanych projektów
oraz rankingu ukończonej uczelni.

### Jak działa
1. Pobiera wszystkich developerów z grafu Neo4j (`Person`)
2. Zlicza liczbę projektów przypisanych do developera (`WORKED_ON`)
3. Pobiera dane o uczelni developera (`STUDIED_AT → University`)
4. Oblicza score kandydata na podstawie:
   - lat doświadczenia (`years_experience`)
   - liczby projektów
   - rankingu uczelni
5. Sortuje developerów według score i zwraca Top N wyników


Pytanie: Give me best developers based on project counts  

Opd: Here are the best developers based on project counts:  

Victor Cook  

Years of Experience: 5  

Project Count: 5  

University: University of New Justinville  

Ryan Hopkins  

Years of Experience: 5  

Project Count: 5  

University: University of South Brianna  

Darius Villa  

Years of Experience: 7  

Project Count: 4  

University: Not specified  

Mary Hines  

Years of Experience: 5  

Project Count: 4  

University: Not specified  

Mark Carroll  

Years of Experience: 5  

Project Count: 3  

University: University of Adamsbury  

If you need more information or further assistance, feel free to ask!  
