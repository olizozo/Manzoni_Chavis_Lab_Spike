

# 🔬 Pipeline Expert : Excitabilité & Propriétés Intrinsèques
**Manzoni Lab | Neurobiologie de la Plasticité Synaptique**

Ce pipeline est un environnement analytique haute résolution dédié à la caractérisation automatique des propriétés électrophysiologiques membranaires à partir de fichiers d'enregistrement `.abf` (Current-Clamp).

---

## 1. Bases Biophysiques & Signification Physiologique

L'excitabilité intrinsèque est le déterminant majeur de la fonction computationnelle du neurone. Elle définit la manière dont les entrées synaptiques (analogiques) sont converties en décharges de potentiels d'action (numériques).

### A. Le Potentiel de Repos ($V_{rest}$)
Le $V_{rest}$ reflète l'état d'équilibre électrochimique de la cellule, principalement maintenu par les canaux potassiques de fuite (*Leak channels*) et la pompe $Na^+/K^+$-ATPase. Une dépolarisation du $V_{rest}$ peut signaler un état de stress cellulaire ou une réduction des conductances potassiques basales.

### B. Résistance d'Entrée ($R_{in}$)
La $R_{in}$ est inversement proportionnelle à la densité totale des canaux ioniques ouverts au repos.
* **Signification :** Un neurone avec une $R_{in}$ élevée est plus "excitable" car une faible injection de courant suffira à générer une variation de tension importante ($\Delta V = R \cdot \Delta I$).
* **Contexte :** Dans les modèles de neuro-adaptation, une baisse de $R_{in}$ est souvent corrélée à une hypertrophie de l'arborisation dendritique (loi de surface).

### C. Capacitance Membranaire ($C_m$) et Constante de Temps ($\tau_m$)
La membrane lipidique se comporte comme un condensateur ($C_m$) en parallèle avec une résistance ($R_{in}$).
* **$\tau_m$ :** Définit la "mémoire" électrique du neurone. Un $\tau_m$ long favorise la sommation temporelle des entrées synaptiques distantes.
* **$C_m$ :** C'est un marqueur direct de la surface membranaire totale ($~1 \mu F/cm^2$). Une augmentation de $C_m$ traduit une croissance physique du neurone.



### D. Courant $I_h$ (Sag)
L'affaissement du voltage (*Sag*) lors d'une hyperpolarisation est la signature des canaux HCN. Ce courant "pacemaker" régule l'excitabilité sous-liminaire et la résonance du neurone.

---

## 2. Formalisme Mathématique & Algorithmes

Le pipeline utilise des méthodes de régression et de dérivation numérique pour garantir la reproductibilité des mesures.

### Extraction de la Résistance d'Entrée ($R_{in}$)
Pour éviter les non-linéarités induites par l'activation de conductances voltage-dépendantes, $R_{in}$ est calculée exclusivement sur le régime passif :
1.  Sélection des 4 premiers échelons hyperpolarisants ($I < 0$).
2.  Extraction du voltage stationnaire ($V_{ss}$) en fin de pulse.
3.  **Régression linéaire :** La pente de la droite $V_{ss} = f(I_{inj})$ donne $R_{in}$.
    * *Note :* Si l'unité est le nA, la pente est directement en $M\Omega$. Si l'unité est le pA, le résultat est multiplié par 1000.

### Détermination du Seuil de Tension ($V_{threshold}$)
Contrairement aux méthodes classiques par seuil fixe, ce pipeline utilise la **cinétique de phase** :
* L'algorithme calcule la dérivée première du voltage : $\frac{dV}{dt} = \frac{V_{n+1} - V_n}{\Delta t}$.
* Le point de seuil est défini comme l'instant où l'accélération dépasse une valeur critique (par défaut $15 \, mV/ms$). C'est le moment biophysique où le courant sodique entrant ($I_{Na}$) surpasse les courants de fuite.



### Calcul de la Capacitance ($C_m$)
En Current-Clamp, $C_m$ est dérivée de la cinétique de charge :
$$\tau_m = R_{in} \cdot C_m \implies C_m = \frac{\tau_m}{R_{in}}$$
Le pipeline mesure $\tau_m$ sur le pulse hyperpolarisant le plus faible pour minimiser l'impact du courant $I_h$. $\tau_m$ est défini par le temps nécessaire pour atteindre $1 - 1/e$ ($~63.2\%$) de l'amplitude totale de la réponse.

---

## 3. Guide d'Utilisation du Pipeline

### Configuration Initiale
1.  **Unité de Courant :** Vérifiez si votre protocole pClamp enregistre en pA ou nA. Un mauvais choix faussera le calcul de $R_{in}$ d'un facteur 1000.
2.  **Seuil dV/dt :** Si vos potentiels d'action ont une phase d'ascension lente, abaissez ce seuil à $10 \, mV/ms$. Pour des neurones très rapides (interneurones), montez à $20 \, mV/ms$.

### Exploration Visuelle
* **Points Rouges :** Ils permettent de synchroniser la trace visualisée en haut avec sa position sur les courbes I-V et f-I en bas. Cela permet d'identifier instantanément quel sweep a généré une valeur atypique.
* **Overlay :** Utilisez cette fonction pour comparer la rhéobase visuellement ou pour observer l'évolution du Sag sur plusieurs intensités.

### Exportation
Le bouton **"Exportation des Résultats"** génère deux fichiers :
1.  `_biophysique.csv` : Résumé des constantes pour la cellule (Tableau prêt pour les statistiques).
2.  `_donnees_courbes.csv` : Points bruts pour reconstruire les courbes I-V et f-I sous Prism.

