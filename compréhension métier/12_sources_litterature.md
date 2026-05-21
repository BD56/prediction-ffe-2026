# Sources de littérature consultées

Document de référence des **sources externes** consultées au cours du projet : URL, contexte de consultation, ce qu'on en retire pour notre méthodologie.

Sert à :
1. Tracer les références dans le rapport final
2. Pouvoir y revenir pour creuser un point
3. Documenter ce qui a été utilisé vs ce qui reste à explorer

---

## 1. Méthodologie générale -- BLUP et évaluation génétique équine

### Anne Ricard (IFCE-INRAE) -- évaluation génétique des chevaux français

| Élément | Info |
|---|---|
| **URL principale** | [Anne Ricard - INRAE Toulouse](https://genphyse.toulouse.inra.fr/people/ricard/anne) |
| **Médiathèque IFCE** | [Bibliographie Ricard](https://mediatheque.ifce.fr/index.php?lvl=author_see&id=3124&page=1&nbr_lignes=162) |
| **Ce qu'on en retient** | Référence française majeure sur l'évaluation génétique des chevaux de sport. Utilise BLUP depuis 1986 pour l'indice ISO (CSO). Le phénotype standard est `log(GAINS + 1)` standardisé. Corrige pour effets environnementaux : saison, sexe, âge. |
| **Application projet** | (a) Convention de transformation `log(GAINS+1)` pour Famille 2. (b) Principe d'effet aléatoire pour le cavalier (Famille 7). |
| **Limite** | Ne détaille pas publiquement le découpage exact des races dans ses modèles. |

### Indices IFCE (ISO, ICC, IDR)

| Élément | Info |
|---|---|
| **URL** | [Indices CSO/CCE/Dressage IFCE](https://equipedia.ifce.fr/elevage-et-entretien/genetique/selection-et-indices/indices-chevaux-cso-cce-et-dressage) |
| **Ce qu'on en retient** | Indices BLUP officiels pour les chevaux français : ISO (saut d'obstacles), ICC (concours complet), IDR (dressage). Publiés depuis 1986 pour CSO, 1997 pour CCE/dressage. Calculés sur toutes les performances + pedigree. |
| **Application projet** | Référence à mentionner dans le rapport comme "modèle de référence" pour la France. Notre approche est complémentaire (prédiction de hauteur, pas BLUP génétique). |

### Performances et BLUP : orientations nouvelles -- Ricard

| Élément | Info |
|---|---|
| **URL** | [Notice médiathèque IFCE](https://mediatheque.ifce.fr/index.php?lvl=notice_display&id=3370) |
| **Ce qu'on en retient** | Document IFCE discutant les évolutions méthodologiques BLUP. |

---

## 2. Race et regroupement -- pratique académique

### "Efficiency of past selection of the French Sport Horse: Selle Français breed" -- Ricard et al.

| Élément | Info |
|---|---|
| **URL ScienceDirect** | [Article ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1871141307001990) |
| **URL ResearchGate** | [ResearchGate](https://researchgate.net/publication/248565599_Efficiency_of_past_selection_of_the_French_Sport_Horse_Selle_Francais_breed_and_suggestions_for_the_future) |
| **Ce qu'on en retient** | Étude BLUP du Selle Français de 1974 à 2002. **Trend annuel** : 0,055 σ_génétique 1985-1995, 0,096 depuis 1995. **Héritabilité** SO : 0,05 à 0,28. |
| **Citation clé** | "Heritability estimates of 0.15 and 0.11, without and with breed groups in the model, indicate that breed group effects can meaningfully influence genetic models" |
| **Application projet** | Confirme que le **regroupement de races en effets fixes** est une pratique standard. Justifie notre approche en Famille 10. |

### "The BLUP-animal model for the estimation of the breeding value of show jumping horses"

| Élément | Info |
|---|---|
| **URL ResearchGate** | [Article ResearchGate](https://www.researchgate.net/publication/289639864_The_BLUP-animal_model_for_the_estimation_of_the_breeding_value_of_show_jumping_horses) |
| **Ce qu'on en retient** | Méthodologie BLUP-animal model pour le saut d'obstacles. **Confirme** : "BLUP models incorporating breed group effects as fixed effects are standard practice in equine genetic evaluation". |
| **Application projet** | Justifie le principe de regroupement par "breed group" dans nos features. |

### Stud-book Selle Français -- règlement officiel

| Élément | Info |
|---|---|
| **URL Wikipedia** | [Selle Français Wikipedia](https://en.wikipedia.org/wiki/Selle_Fran%C3%A7ais) |
| **URL IFCE** | [Règlement Stud-Book SF (IFCE)](https://www.ifce.fr/document/reglement-stud-book-selle-francais/) |
| **URL Cheval Grand Est** | [PDF Règlement stud-book SF](https://www.cheval-grandest.com/userfiles/document/20221021133339-) |
| **Ce qu'on en retient** | Le Selle Français est créé en 1958 par **fusion de plusieurs races françaises**. Le studbook accepte des reproducteurs Pur-Sang, Arabe, Anglo-Arabe, Trotteur français → c'est déjà un "super-groupe" mélangeant des origines, pas une race "pure" au sens biologique. |
| **Application projet** | Point important pour le rapport : le regroupement "Selle Français" qu'on propose recoupe déjà plusieurs origines. Cohérence métier. |

### WBFSH (World Breeding Federation for Sport Horses) -- classification internationale

| Élément | Info |
|---|---|
| **URL principale** | [WBFSH](https://wbfsh.com/) |
| **Rankings** | [WBFSH Rankings](https://wbfsh.com/rankings) |
| **Sire Rankings** | [WBFSH Sire Rankings](https://wbfsh.com/sire-rankings) |
| **Membres / studbooks** | [Studbook Members](https://www.wbfsh.com/studbook-members) |
| **Wikipedia** | [WBFSH Wikipedia](https://en.wikipedia.org/wiki/World_Breeding_Federation_for_Sport_Horses) |
| **Ce qu'on en retient** | La WBFSH classe les chevaux par **studbook** (Selle Français, KWPN, Holsteiner, BWP, Zangersheide, Hannoverian, Oldenburger, etc.), pas par "race" au sens strict. Ranking calculé via résultats FEI sur cycle annuel 1er oct → 30 sept. |
| **Application projet** | Alternative au regroupement "race" : classification par **studbook** (= structure d'élevage). Plus proche de la pratique internationale officielle. |

### Sanchez-Guerrero -- Spanish Sport Horse populations

| Élément | Info |
|---|---|
| **URL ScienceDirect** | [Article ScienceDirect 2017](https://www.sciencedirect.com/science/article/abs/pii/S1871141317300999) |
| **MDPI 2024** | [Integrating Performance Records and Genetic Evaluations in Spanish Horse Populations](https://www.mdpi.com/2075-1729/16/3/455) |
| **Ce qu'on en retient** | Évalue les populations sportives espagnoles, dont le **Caballo de Deporte Español** (CDE) qui montre les EBV les plus élevés en SO. Examine plusieurs studbooks. Approche similaire au principe "regroupement par studbook". |
| **Application projet** | Référence pour justifier l'inclusion de plusieurs populations / studbooks dans un même modèle. |

### Viklund -- Swedish Warmblood

| Élément | Info |
|---|---|
| **Référence** | Heritability estimates pour SWB : 0,12 à 0,28 en CSO |
| **Ce qu'on en retient** | Approche Swedish Warmblood (SWB), avec young horse tests comme indicateurs. Pas d'info trouvée sur regroupement multi-races. |

---

## 3. Acteurs commerciaux et indices industriels

### EquiRatings

| Élément | Info |
|---|---|
| **Méthodologie connue** | Système ELO basé sur volume + clear rounds (= sans-faute). Utilisé dans le sport irlandais et international. |
| **Ce qu'on en retient** | Métrique "Clear Rounds" comme phénotype principal. Approche dynamique (ELO) plutôt que statique. |
| **Application projet** | Inspiration pour la métrique "sans-faute" qu'on a finalement écartée (Famille 3) faute de doc FFE sur les barèmes. |

### Hippomundo Rating

| Élément | Info |
|---|---|
| **URL** | [Hippomundo](https://www.hippomundo.com/) |
| **Méthodologie** | Combine niveau d'épreuve + plateau atteint + fréquence sans-faute |
| **Ce qu'on en retient** | Approche multi-dimensionnelle de la performance. Nous a inspiré la liste des familles de features. |

### Horsetelex ISV / IPV

| Élément | Info |
|---|---|
| **Méthodologie** | Indice basé sur Gains / nb départs |
| **Ce qu'on en retient** | Confirme l'intérêt du **ratio gains/participations** comme indicateur de qualité (capté dans Famille 2 avec `gains_par_participation_4_7`). |

### JPR (Jumper Performance Rating, USA)

| Élément | Info |
|---|---|
| **Méthodologie** | Combine volume, placement, gains, consistance |
| **Ce qu'on en retient** | Confirme les 4 familles de base : Activité, Placement, Gains, Consistance (= variabilité). |

---

## 4. Études spécifiques mentionnées

### Léa Chapard (INRAE) -- chercheuse en génétique équine

**Affiliation** : Université Paris-Saclay, INRAE, AgroParisTech, GABI (Jouy-en-Josas, France)

**Domaine** : génétique équine, évaluation BLUP, traits de jeunesse en saut d'obstacles.

### Chapard 2023 -- "Adjusted fence height" ✅

| Élément | Info |
|---|---|
| **Titre exact** | "Adjusted fence height: an improved phenotype for the genetic evaluation of show jumping performance in Warmblood horses" |
| **Journal** | Genetics Selection Evolution |
| **URL** | [Article complet](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-023-00786-2) |
| **Contenu** | Développement d'une méthode pour ajuster la hauteur d'obstacle en fonction du classement et du niveau de compétition. Phénotype amélioré pour l'évaluation génétique BLUP en SO. |
| **Statut** | ✅ **CONFIRMÉ par recherche web** |
| **Application projet** | Inspiration de la métrique "hauteur d'obstacle ajustée" qu'on a évoquée. **Mais on a finalement rejeté toute la Famille 4 (Hauteurs)** -- l'approche Chapard reste une référence à mentionner dans le rapport comme alternative possible. |

### Chapard et al. -- "Accelerometers as Genetic Selection Criteria" 2020 ✅

| Élément | Info |
|---|---|
| **Titre** | "Accelerometers Provide Early Genetic Selection Criteria for Jumping Horses" |
| **URL** | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32508876/) |
| **Contenu** | 1 056 chevaux de 3 ans équipés d'accéléromètres 3D pendant des tests de saut libre 2015-2017. Démontre la pertinence des capteurs pour la sélection génétique précoce. |
| **Statut** | ✅ Confirmé |
| **Application projet** | Approche différente (mesures cinématiques avec capteurs) -- pas applicable directement à notre base FFE, mais valide le principe "early selection criteria for jumping horses". |

### Traits de jeunesse → performance (Belgian Warmblood) ✅

| Élément | Info |
|---|---|
| **Statut** | ✅ Confirmé (référencé via recherche web) |
| **Citation correcte** | Pour les traits linéaires de jeunesse en CSO : **héritabilités 0,04-0,38** ; **corrélations génétiques 0,40 à 0,65** avec la performance sportive adulte (Belgian Warmblood). |
| **Erreur précédente** | J'avais cité "corrélations 0,30 à 0,77" -- chiffres inexacts. Les vrais chiffres sont 0,40-0,65. |
| **Application projet** | Justifie scientifiquement le principe que des traits observés tôt (notre fenêtre 4-7 ans) sont prédictifs de la performance adulte. Référence importante pour le rapport. |

### Ricard & Blouin 2011 -- Survival analysis sur longévité

| Élément | Info |
|---|---|
| **Statut** | ⚠ Référence présumée -- non re-vérifiée individuellement, mais Anne Ricard est confirmée comme chercheuse IFCE active. Référence à valider individuellement avant citation. |
| **Ce qui est présumé** | Analyse de survie sur la longévité de carrière avec features démographiques. |

---

## 5. Réglementation FFE / SHF

### Règlement CSO FFE 2026

| Élément | Info |
|---|---|
| **URL PDF** | [Règlement CSO FFE 2026 (PDF)](https://www.ffe.com/system/files/disciplines/reglement-cso-2026-version-de-travail-avec-rectificatif-applicable-au-05-01-2026.pdf) |
| **Ce qu'on en retient** | Règles d'élimination, barèmes (sans description de la codification informatique des statuts). |
| **Application projet** | Confirmation des règles métier (3 refus = élimination, etc.) sans accès aux codes informatiques. |

### Règlement CSO 2023

| Élément | Info |
|---|---|
| **URL PDF** | [Règlement CSO FFE 2023](https://www.ffe.com/system/files/disciplines-cso/reglements/REGLEMENT_CSO_2023_applicable_au_01.09.2022_4.pdf) |

### Classements en Saut d'obstacles (FFE)

| Élément | Info |
|---|---|
| **URL** | [Classements SO FFE](https://www.ffe.com/competition/circuits-et-championnats/circuits-ffe/grand-national/cso/recapitulatif-des-classements) |
| **Ce qu'on en retient** | Système de points et classements officiels FFE. |

### SHF -- Règlement Jeunes Chevaux CSO

| Élément | Info |
|---|---|
| **URL principale** | [Règlements SHF Jeunes Chevaux CSO](https://www.shf.eu/fr/valorisation/informations-documents/cso-chevaux/reglements.html) |
| **URL PDF 2023** | [Règlement SHF CSO 2023 (PDF)](https://www.shf.eu/userfiles/valorisation/cso_chevaux/2023/reglements/2023_reglement_cso_cc_et_cl_2023-31-01.pdf) |
| **URL Aide-mémoire** | [Cycle Classique SHF](https://www.shf.eu/userfiles/valorisation/cso_chevaux/2023/aides_memoires/shf2023_aidememoireparcours_cc.pdf) |
| **Ce qu'on en retient** | Règlement des cycles SHF jeunes chevaux 4-7 ans (Cycle Libre, Cycle Classique, Label, Formation). Mentionne les éliminations mais pas les hauteurs réglementaires détaillées dans un format utilisable. |
| **Application projet** | Confirmation du parcours formateur SHF mais sans accès à la grille des hauteurs (= notre demande FFE non aboutie). |

### Stud-book Anglo-Arabe -- AVL Genetics

| Élément | Info |
|---|---|
| **URL** | [AVL Genetics - Stud books](https://avlgenetics.aveyron-labo.com/fr/infos-faq/equins-exigences-des-stud-books/) |
| **Ce qu'on en retient** | Détails sur les exigences des stud-books français incluant Anglo-Arabe. |

---

## 6. Forum / discussions équestres (statuts non-classants)

### Forum cheval-annonce : différence entre disqualification, élimination, abandon

| Élément | Info |
|---|---|
| **URL** | [Forum cheval-annonce](https://www.chevalannonce.com/forums-6591440-difference-entre-disqualification-eliminination-abandon) |
| **Ce qu'on en retient** | Définitions claires des statuts non-classants (non-partant, abandon, élimination, disqualification, hors concours, non-classé). Pas de codes numériques mentionnés. |
| **Application projet** | Aide à comprendre la sémantique des statuts, mais aucun mapping avec les codes 899/900/902/992/993 de notre base. |

---

## 7. Statut récap des sources

| Sujet | Sources fiables ? | Application validée ? |
|---|---|---|
| BLUP général | ✅ Multiple | ✅ Référence dans rapport |
| Effet "breed group" dans BLUP | ✅ Confirmé (Ricard et al., articles BLUP) | ✅ Famille 10 |
| Découpage exact des races | ⚠ Pas standardisé internationalement | À documenter comme choix méthodologique |
| Anne Ricard / IFCE | ✅ Confirmée | ✅ Inspiration log(GAINS+1) |
| WBFSH (classification par studbook) | ✅ Confirmée | Alternative possible à notre découpage |
| Codes PLACE FFE (899, 900, ...) | ❌ Non documenté publiquement | Traités en bloc (cf. catalogue) |
| Chapard 2023 (Adjusted fence height) | ✅ **CONFIRMÉE** (Léa Chapard, INRAE) | Citable comme référence ferme |
| Chapard 2020 (Accelerometers) | ✅ Confirmée | Citable comme référence ferme |
| Traits de jeunesse → performance | ✅ Confirmé (corrélations 0,40-0,65) | Citable -- corriger les chiffres |
| Sanchez-Guerrero | ✅ Confirmée | Inspiration multi-populations |
| Viklund | ✅ Confirmée | Référence Swedish Warmblood |
| EquiRatings / Hippomundo / Horsetelex / JPR | ✅ Existent | Inspirations conceptuelles, pas citations directes |

---

## 8. Pistes restantes à explorer (si besoin)

- ~~**Chapard** : tenter de retrouver la vraie référence~~ → **Résolu 2026-05-10** : Léa Chapard (INRAE), références confirmées.
- **Grille hauteurs SHF** : on attendait la réponse FFE -- non aboutie. Reste un manque méthodologique documenté.
- **Codes PLACE FFE** : portail technique `toutsavoir.ffecompet.com` inaccessible (403). Pas de doc publique.
- **Ricard & Blouin 2011** (survival analysis) : référence présumée à vérifier individuellement avant citation.

---

## 9. Notes sur les vérifications de littérature

Les recherches du 2026-05-10 ont permis de :
- ✅ **Confirmer Léa Chapard** comme auteur réel des références sur la "hauteur d'obstacle ajustée" et les "traits de jeunesse"
- ✅ **Corriger les chiffres** sur les corrélations génétiques (réels : 0,40-0,65, et non 0,30-0,77)
- ✅ **Confirmer le principe** "breed group effect comme effet fixe" en BLUP
- ⚠ **Identifier** que le découpage exact des races n'est pas standardisé internationalement

---

**Document créé le 2026-05-10 -- à compléter au fur et à mesure des recherches.**
