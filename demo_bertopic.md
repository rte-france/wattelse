# Démo Bertopic sur le thème de la décarbonation

## Qu'est-ce que la décarbonation?
(merci ChatGPT!)

Dans le domaine électrique, la décarbonation vise principalement à réduire les émissions de dioxyde de carbone (CO2) générées par la production, la distribution et la consommation d'électricité. Voici quelques concepts clés liés à la décarbonation dans ce domaine :
-  Énergies renouvelables : L'intégration croissante d'énergies renouvelables telles que l'énergie solaire, éolienne, hydraulique et géothermique dans le mix énergétique permet de produire de l'électricité sans émissions de CO2.
-  Transition énergétique électrique : Cette transition implique le remplacement progressif des sources d'énergie fossile, comme le charbon et le gaz naturel, par des sources d'énergie propres et renouvelables dans la production d'électricité.
-  Stockage d'énergie : Le développement de technologies de stockage d'énergie, telles que les batteries et les systèmes de stockage à l'hydrogène, permet de mieux gérer l'intermittence des sources d'énergie renouvelables et de fournir de l'électricité lorsqu'elle est nécessaire, contribuant ainsi à la décarbonation.
-  Réseaux électriques intelligents (smart grids) : Les réseaux électriques intelligents intègrent des technologies de communication et de contrôle avancées pour optimiser la production, la distribution et la consommation d'électricité, facilitant ainsi l'intégration des énergies renouvelables et la réduction des émissions.
-  Électrification des secteurs : L'électrification de secteurs tels que les transports (véhicules électriques), le chauffage et la climatisation (pompes à chaleur) permet de déplacer la demande d'énergie vers l'électricité propre, réduisant ainsi les émissions globales.
-  Cogénération et trigénération : Ces concepts consistent à produire simultanément de l'électricité et de la chaleur (ou de l'électricité, de la chaleur et du froid) à partir d'une seule source d'énergie, améliorant ainsi l'efficacité énergétique globale et réduisant les émissions.
-  Amélioration de l'efficacité des centrales thermiques : L'optimisation des technologies de production d'électricité à partir de combustibles fossiles peut réduire les pertes d'énergie et les émissions associées.
-  Micro-réseaux et production décentralisée : La mise en place de micro-réseaux et de systèmes de production d'électricité décentralisée, tels que les panneaux solaires sur les toits et les éoliennes locales, peut contribuer à la production d'électricité propre à petite échelle.
-  Numérisation et automatisation : L'utilisation de technologies numériques et d'automatisation permet d'optimiser la gestion des flux d'électricité, d'améliorer la coordination des sources d'énergie et de réduire les pertes d'énergie, contribuant ainsi à la décarbonation.
- Normes et réglementations : Les normes et les réglementations gouvernementales peuvent encourager l'adoption de technologies à faible émission de carbone, favorisant ainsi la décarbonation du secteur électrique.

Ces concepts interagissent pour créer un écosystème électrique plus propre et plus durable, tout en contribuant à la réduction des émissions de CO2.


## Définition de requêtes pour trouver des news liées à la décarbonation

### Tests de requêtes
décarbonation
"énergies renouvelables"
"transition énergétique électrique"
"réseaux électriques intelligents"
"smart grids" 
"stockage d'énergie"
"stockage d'électricité"
"efficacité énergétique"
cogénération
trigénération

décarbonation OR "énergies renouvelables" OR "transition énergétique électrique" OR "réseaux électriques intelligents" OR "smart grids" OR "stockage d'énergie" OR "stockage d'électricité" OR "efficacité énergétique" OR cogénération OR trigénération

## Génération de datasets

### Génération des requêtes
**/!\ si la requête Google news est trop longue, les termes de date début / fin sont ignorés !!**

Par exemple, avec:
https://news.google.com/search?q=d%C3%A9carbonation+OR+%22%C3%A9nergies+renouvelables%22+OR+%22transition+%C3%A9nerg%C3%A9tique+%C3%A9lectrique%22+OR+%22r%C3%A9seaux+%C3%A9lectriques+intelligents%22+OR+%22smart+grids%22+OR+%22stockage+d%27%C3%A9nergie%22+OR+%22stockage+d%27%C3%A9lectricit%C3%A9%22+OR+%22efficacit%C3%A9+%C3%A9nerg%C3%A9tique%22++OR+cog%C3%A9n%C3%A9ration+OR+trig%C3%A9n%C3%A9ration+after:2018-01-01+before:2018-02-01&ceid=FR:fr&hl=fr&gl=FR
ne tient pas compte des dates spécifiées.

Empiriquement, se limiter à une requête de la taille de l'exemple suivant:
(au besoin la découper en 2)

`décarbonation OR "énergies renouvelables" OR "transition énergétique électrique" OR "smart grids" OR "stockage d'énergie" OR "efficacité énergétique" OR cogénération after:2018-01-01 before:2018-02-01

Génération des fichiers requêtes:
- pas hedbomadaire
- 50 résultats max par pas
 
Exécuter la commande suivante
```
PYTHONPATH=. python -m data_provider generate-query-file "décarbonation OR \"énergies renouvelables\" OR \"transition énergétique électrique\" OR \"smart grids\" OR \"stockage d'énergie\" OR \"efficacité énergétique\" OR cogénération" --after 2018-01-01 --before 2023-07-31 --save-path decarbonation.txt --interval 7
```


### Collection de données Google
Lancer la commande suivante:
```
PYTHONPATH=. python -m data_provider auto-scrape data_provider/query_examples/decarbonation.txt --max-results 100 --save-path decarbonation.jsonl
```

Compter environ 1h30 pour récupérer ~20000 documents.

## Exploration des données collectées

26572 documents
