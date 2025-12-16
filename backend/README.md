## Algorytm dopasowania -> tools_matching.py

Narzędzie match_devs_to_rfp_simple w najprostszy sposób dopasowuje developerów do jednego RFP. Na podstawie relacji NEEDS pobiera  

wymagane umiejętności (oraz liczbę osób na skill), a następnie wyszukuje w Neo4j osoby posiadające dane umiejętności (HAS_SKILL). Każdy  

developer może zostać przypisany tylko raz. Wynikiem jest lista przypisań {skill, personId, name} oraz lista braków (unfilled) dla  

umiejętności, dla których nie znaleziono wystarczającej liczby osób. Sortuje kandydatów po ilości projektów.



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


## Algorytm rankingu -> tools_ranking.py

Narzędzie rank_best_devs_university służy do rankingu developerów na podstawie łącznego score’u obliczanego z: lat doświadczenia (45%),

liczby projektów (45%) oraz rankingu uczelni (10%). Dane pobierane są z bazy Neo4j, opcjonalnie filtrowane po słowie kluczowym projektu. 

Dla każdej osoby wyliczany jest score ∈ [0,1], po czym wyniki są sortowane malejąco i zwracane jako Top N najlepszych developerów.  

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
