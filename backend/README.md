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
3. Filtruje developerów pod kątem dostępności na dzień startu RFP: odrzuca osoby, które mają przypisanie ASSIGNED_TO nachodzące na Rfp.start_date (tj. a.start_date ≤ start_date i a.end_date jest puste lub a.end_date ≥ start_date)
4. Oblicza score developera na podstawie:
   - lat doświadczenia
   - liczby zrealizowanych projektów
   - rankingu uczelni
5. Sortuje kandydatów według score i wybiera najlepszych
6. Zapobiega przypisaniu jednego developera do wielu umiejętności
7. Zwraca przypisania oraz listę nieobsadzonych wymagań


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


## Tool make_whatif_match_tool

Tool służy do dopasowania developerów do wymagań konkretnego RFP
z wykorzystaniem grafu Neo4j, z możliwością przeprowadzenia analizy
what-if poprzez dodanie hipotetycznych developerów.

Jak działa

Pobiera wymagania RFP z grafu Neo4j (Rfp → NEEDS → Skill)

Dla każdej wymaganej umiejętności:

pobiera wymaganą liczbę developerów (required_count)

Wyszukuje kandydatów z grafu Neo4j:

developerów posiadających daną umiejętność (HAS_SKILL)

filtruje developerów pod kątem dostępności na dzień startu RFP: odrzuca osoby, które mają przypisanie ASSIGNED_TO nachodzące na Rfp.start_date (tj. a.start_date ≤ start_date i a.end_date jest puste lub a.end_date ≥ start_date)

zlicza liczbę projektów (WORKED_ON → Project)

pobiera dane o uczelni (STUDIED_AT → University)

Uwzględnia dodatkowych, hipotetycznych developerów (extra_devs)
przekazanych w analizie what-if

Oblicza score kandydata na podstawie:

lat doświadczenia (yearsExperience)

liczby projektów

rankingu ukończonej uczelni

Sortuje kandydatów według score i przypisuje najlepszych developerów
do każdej wymaganej umiejętności

Zwraca przypisania wraz z informacją o brakach kadrowych


Pytanie: What if we add two extra developers to the RFP "API Integration Platform Development": 
W1 (Anna): skills Java,AWS; 7 years; 10 projects; university ranking 80
W2 (Bartek): skills Python,Docker; 4 years; 6 projects; university ranking 200 Return assignments with scores.

Odp: Here are the assignments with scores for the RFP "API Integration Platform Development" after adding the two extra developers:

Assignments:
Skill: Java

Developer: Anna (W1)
Score: 0.7974
Years Experience: 7
Project Count: 10
University Ranking: 80
Skill: Angular

Developer: Jessica Erickson
Score: 0.3734
Years Experience: 5
Project Count: 2
University Ranking: Not specified
Skill: Django

Developer: Lauren Flowers
Score: 0.1484
Years Experience: 0
Project Count: 2
University Ranking: Not specified
Skill: Docker

Developer: Darius Villa
Score: 0.5628
Years Experience: 7
Project Count: 4
University Ranking: 9999
Skill: Typescript

Developer: Mark Carroll
Score: 0.428
Years Experience: 5
Project Count: 3
University Ranking: N/A
Unfilled Skills:
Angular: Required 3, Filled 1
Django: Required 2, Filled 1
Java: Required 2, Filled 1
Typescript: Required 3, Filled 1
This summary includes the developers assigned to each skill along with their scores and experience details. Let me know if you need further assistance!

## Tool compare_baseline_vs_whatif_for_rfp

Tool służy do porównania wyników dopasowania developerów do RFP
pomiędzy scenariuszem bazowym (tylko dane z grafu Neo4j)
a scenariuszem what-if, w którym uwzględniani są dodatkowi,
hipotetyczni developerzy.

Jak działa

Uruchamia dopasowanie bazowe do RFP

bez uwzględniania dodatkowych developerów (extra_devs = [])

Uruchamia dopasowanie what-if do RFP

z uwzględnieniem developerów przekazanych w extra_devs

Porównuje przypisania z obu scenariuszy:

identyfikuje nowe przypisania (added_assignments)

identyfikuje usunięte przypisania (removed_assignments)

Analizuje zmiany per umiejętność:

porównuje listy przypisanych developerów w scenariuszu bazowym
i what-if

Porównuje braki kadrowe:

wskazuje umiejętności, dla których sytuacja uległa poprawie,
pogorszeniu lub nie uległa zmianie

Zwraca pełne wyniki scenariusza bazowego i what-if
wraz z podsumowaniem różnic

Pytanie: Compare baseline vs what-if for the RFP "API Integration Platform Development" after adding two extra developers
W1 (Anna): skills Java,AWS; 7 years; 10 projects; university ranking 80
W2 (Bartek): skills Python,Docker; 4 years; 6 projects; university ranking 200 Return assignments with scores.


Odp: Here's the comparison of the baseline vs. what-if for the RFP "API Integration Platform Development" after adding the two extra developers, Anna and Bartek.

Baseline Assignments
Skill: Angular

Developer: Jessica Erickson
Score: 0.3734
Years Experience: 5
Project Count: 2
University Ranking: Not specified
Skill: Django

Developer: Lauren Flowers
Score: 0.1484
Years Experience: 0
Project Count: 2
University Ranking: Not specified
Skill: Docker

Developer: Darius Villa
Score: 0.5628
Years Experience: 7
Project Count: 4
University Ranking: 9999
Skill: Typescript

Developer: Mark Carroll
Score: 0.428
Years Experience: 5
Project Count: 3
University Ranking: N/A
What-If Assignments
Skill: Angular

Developer: Jessica Erickson
Score: 0.3734
Years Experience: 5
Project Count: 2
University Ranking: Not specified
Skill: Django

Developer: Lauren Flowers
Score: 0.1484
Years Experience: 0
Project Count: 2
University Ranking: Not specified
Skill: Docker

Developer: Darius Villa
Score: 0.5628
Years Experience: 7
Project Count: 4
University Ranking: 9999
Skill: Java

Developer: Anna (W1)
Score: 0.7974
Years Experience: 7
Project Count: 10
University Ranking: 80
Skill: Typescript

Developer: Mark Carroll
Score: 0.428
Years Experience: 5
Project Count: 3
University Ranking: N/A
Differences
Added Assignments:

Skill: Java
Developer: Anna (W1)
Score: 0.7974
Removed Assignments: None

Per Skill Changes:

Skill: Java
Baseline: No assignments
What-If: Anna assigned with a score of 0.7974
Unfilled Skills
Baseline:

Angular: Required 3, Filled 1
Django: Required 2, Filled 1
Java: Required 2, Filled 0
Typescript: Required 3, Filled 1
What-If:

Angular: Required 3, Filled 1
Django: Required 2, Filled 1
Java: Required 2, Filled 1
Typescript: Required 3, Filled 1
Summary of Unfilled Changes
Improved:

Skill: Java
Baseline missing: 2
What-If missing: 1
Unchanged:

Angular, Django, and Typescript remained the same in terms of filled assignments.
This analysis shows that adding Anna significantly improved the assignment for the Java skill, while other skills remained unchanged in terms of filled assignments.