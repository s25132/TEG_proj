# TEG_proj
## Skrypt generate_data -> katalog generate_data
Służy do generowania cv programistów, rfps i pliku projects.json z obecnie toczącymi się projektami.  

Pliki sa generowane do katalgu /data w gałęzi głownej


Uruchomienie:  

generate_data_docker_example.yml zmień na generate_data_docker.yml

uzupełnij OPENAI_API_KEY w generate_data_docker.yml 

będąc w katalogu głownym:

docker compose -f generate_data_docker.yml build 

docker compose -f generate_data_docker.yml up

Istniejące dane zostaną usunięte


## Skrypt create_chroma -> katalog create_dbs
Służy do stworzenia chromadb na podstaawie cv programistów i pliku projects.json z obecnie toczącymi się projektami.  

Pliki sa generowane do katalgu /data w gałęzi głownej.  

Uruchomienie:  

create_dbs_docker_example.yml zmień na create_dbs_docker.yml

uzupełnij OPENAI_API_KEY w create_dbs_docker.yml 

będąc w katalogu głownym:

docker compose -f create_dbs_docker.yml build 

docker compose -f create_dbs_docker.yml up

Istniejące dane zostaną usunięte




