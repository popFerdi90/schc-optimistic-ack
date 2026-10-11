# Note explicative — publier ce dossier comme dépôt GitHub

Cette note décrit, pas à pas, comment transformer le dossier `schc-optimistic-ack/`
en dépôt GitHub contenant tout ce qu'il faut pour **refaire les expériences et obtenir
les mêmes figures** que dans l'article. Elle n'est pas destinée à rester dans le dépôt
public : vous pouvez la supprimer après usage (`git rm GITHUB_SETUP.md`).

## 1. Ce qui est dans le dépôt, et ce qui n'y est pas

| Inclus | Rôle |
|---|---|
| `simulator/`, `results/`, `figs/`, `audit_tables.py`, `scripts/`, `tests/` | code, résultats de référence, figures, vérifications |
| `sn-article.tex`, `sn-article.pdf`, `sn-bibliography.bib`, `sn-jnl.cls`, `sn-basic.bst` | manuscrit compilable |
| `README.md`, `Makefile`, `requirements.txt`, `requirements-lock.txt`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.github/workflows/ci.yml` | mode d'emploi, dépendances, licence, citation, intégration continue |

| Volontairement exclus de l'archive d'origine | Pourquoi |
|---|---|
| `reponse_relecteur*.tex/.pdf` | lettres aux relecteurs : confidentielles tant que la revue est en cours, et elles citent d'anciennes numérotations |
| `cs-*.png`, `fig.eps`, `empty.eps`, `user-manual.pdf`, `figs/chap*_POO/`, `figs/AASS2.png`, `figs/fig1.png`, `figs/fig8.png`, `preamble/` | restes du gabarit ou d'autres documents, non utilisés par l'article |
| `bst/` (7 styles non utilisés), `REDUCTION_V3.md`, `REDUCTION_V4.md` | notes de travail internes ; seul `sn-basic.bst` est utilisé |
| `__pycache__/`, fichiers `.aux/.log/.bbl…` | produits de compilation |

## 2. Préparer et vérifier en local (avant toute publication)

```bash
cd schc-optimistic-ack
make venv        # crée .venv et installe requirements-lock.txt
make test        # quelques secondes
make verify      # rejoue TOUTE la campagne dans .verify/ et compare avec results/
make paper       # (optionnel) recompile sn-article.pdf
```

Résultat attendu : `make test` → `OK` ; `make verify` → `IDENTICAL`.
Sans `make` (Windows) : voir la section « No make? » du `README.md`.

## 3. Créer le dépôt GitHub

**Nom conseillé :** `schc-optimistic-ack` (court, explicite). **Visibilité :**

* tant que l'article est en relecture, vérifiez la politique de la revue
  (relecture en simple ou double aveugle). Si l'anonymat est exigé, créez le dépôt en
  **privé** et donnez un accès temporaire, ou utilisez un service d'anonymisation
  (par exemple *Anonymous GitHub*) ; les noms des auteurs figurent dans `LICENSE`,
  `CITATION.cff` et `sn-article.tex`, à retirer dans ce cas ;
* sinon, **public**.

### Option A — interface web

1. github.com → **New repository**.
2. *Repository name* : `schc-optimistic-ack` ; *Description* : « Simulator and
   reproduction package for Adaptive SCHC ACK-on-Error (LoRaWAN) ».
3. **Ne cochez pas** « Add a README / .gitignore / license » (ils existent déjà : cela
   créerait un conflit au premier push). Cliquez **Create repository**.

### Option B — ligne de commande (`gh`)

```bash
gh auth login
gh repo create <utilisateur-ou-organisation>/schc-optimistic-ack --private \
   --description "Simulator and reproduction package for Adaptive SCHC ACK-on-Error" \
   --source . --remote origin
```

(remplacez `--private` par `--public` si besoin).

## 4. Premier commit et push

```bash
cd schc-optimistic-ack
git init -b main
git add .
git status                      # contrôle : pas de .venv/, .verify/, __pycache__/
git commit -m "Initial release: simulator, results, figures, manuscript (v1.0.0)"
git remote add origin https://github.com/<utilisateur-ou-organisation>/schc-optimistic-ack.git
git push -u origin main
```

(Si vous avez utilisé `gh repo create … --source .`, le remote existe déjà : sautez
`git remote add`.) Le dépôt pèse quelques Mo (résultats JSON + PDF) : aucun besoin de
Git LFS.

## 5. Renseigner les emplacements à compléter (obligatoire)

Une fois l'URL connue, remplacez les trois marqueurs :

| Fichier | Marqueur | Par |
|---|---|---|
| `README.md` | `<URL-OF-THIS-REPOSITORY>` (section *Quick start*) | l'URL de clonage |
| `CITATION.cff` | `repository-code` (et `doi`, voir §6) | l'URL du dépôt, puis le DOI |
| `sn-article.tex` | `\emph{[Repository URL, archived DOI, release tag and licence to be inserted here before publication.]}` — **2 occurrences** : fin du paragraphe *Reproducibility statement* (§5.1) et *Code availability* (fin d'article) | par exemple : `The artefact is available at \url{https://github.com/<…>/schc-optimistic-ack}, release v1.0.0 (archived at doi:10.5281/zenodo.<…>), under the MIT licence for the code.` |

Puis `make paper`, vérifiez que le PDF reste à 30 pages, et :

```bash
git add -A && git commit -m "Insert repository URL, DOI and licence" && git push
```

(Si `\url` n'est pas disponible avec la classe Springer, utilisez `\texttt{…}`.)

## 6. Archiver une version citable (DOI Zenodo)

1. Sur zenodo.org, connectez-vous avec GitHub, puis **Account → GitHub** et activez
   l'interrupteur du dépôt `schc-optimistic-ack`.
2. Sur GitHub : **Releases → Draft a new release**, *tag* `v1.0.0`, titre
   « v1.0.0 — artefact accompanying the submitted paper », **Publish release**.
3. Zenodo archive la release et attribue un DOI : copiez-le dans `CITATION.cff`
   (`doi:`), dans les deux emplacements du `.tex` et ajoutez un badge au `README.md`.
4. Si la revue demande des corrections de l'article, publiez une nouvelle release
   (`v1.0.1`, …) plutôt que de modifier le tag existant.

## 7. Intégration continue (déjà configurée)

`.github/workflows/ci.yml` s'exécute à chaque push :

* `python -m unittest discover -s tests` (graines, déterminisme, cohérence
  résultats / figure / légende, audit de toutes les cellules des tableaux) ;
* régénération de toutes les figures à partir de `results/`.

Le rejeu complet de la campagne (~15 min) est déclenchable à la main :
**Actions → ci → Run workflow → cocher `full_verify`**.

## 8. Licences — à confirmer par les auteurs

* Le code est sous **MIT** (`LICENSE`, titulaires : les trois auteurs). Si
  l'établissement ou les co-auteurs préfèrent Apache-2.0 ou GPL, remplacez le fichier et
  la ligne `license:` de `CITATION.cff`.
* Le manuscrit et le PDF restent la propriété des auteurs (puis de l'éditeur après
  publication) : vérifiez la politique de dépôt de Springer Nature avant de laisser
  `sn-article.pdf` en accès public ; en cas de doute, retirez-le du dépôt
  (`git rm sn-article.pdf`) et mettez-le en `.gitignore`.
* `sn-jnl.cls` et `sn-basic.bst` sont des fichiers Springer Nature sous leur licence
  propre ; ils sont fournis pour que le manuscrit compile.

## 9. Liste de contrôle avant de rendre le dépôt public

- [ ] `make test` → OK, `make verify` → IDENTICAL (sur votre machine)
- [ ] durée de calcul annoncée dans l'article (« about six minutes on one core », §5.1) cohérente avec votre machine : mesurée ici à ≈ 15 min sur 1 vCPU, à ajuster si besoin
- [ ] URL, DOI et release renseignés (README, CITATION.cff, 2× `sn-article.tex`) ; PDF recompilé
- [ ] aucune lettre aux relecteurs, aucun fichier personnel ou volumineux (`git ls-files`)
- [ ] anonymat respecté si la revue l'exige
- [ ] licences confirmées (§8)
- [ ] onglet **Actions** au vert
- [ ] test à blanc : `git clone` dans un dossier neuf, puis `make venv && make test`

## 10. Ce qu'un tiers fait pour reproduire vos figures

```bash
git clone https://github.com/<utilisateur-ou-organisation>/schc-optimistic-ack.git
cd schc-optimistic-ack
make venv && make test && make verify     # rejeu complet + comparaison, aucune écriture dans results/
make experiments figures                  # (facultatif) régénère results/ puis figs/
```
