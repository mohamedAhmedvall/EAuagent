# CONTRAINTES DU PLAN DE RENOUVELLEMENT — SOMEI (Réseau Eau Potable)

## 1. CONTRAINTES FINANCIÈRES

| Contrainte | Description | Type |
|---|---|---|
| `budget_annuel_max` | Enveloppe budgétaire annuelle maximale | **Dure** |
| `budget_annuel_min` | Dépense minimale annuelle pour maintenir le patrimoine | **Dure** |
| `budget_pluriannuel` | Plafond cumulé sur l'horizon du plan (ex. 5 ans) | **Dure** |
| `cout_km_par_materiau` | Coût unitaire de renouvellement par km selon matériau et diamètre | Paramètre |
| `lissage_budget_pct` | Variation max d'une année sur l'autre (±30%) | **Souple** |

## 2. CONTRAINTES DE CAPACITÉ OPÉRATIONNELLE

| Contrainte | Description | Type |
|---|---|---|
| `km_max_par_an` | Max de km renouvelables par an (capacité chantier/équipes) | **Dure** |
| `km_min_par_an` | Taux de renouvellement annuel cible minimum | **Dure** |
| `chantiers_simultanes_max` | Nb max de chantiers en parallèle (ressources humaines/matériel) | **Dure** |
| `materiau_interdit_maintien` | Matériaux à éliminer obligatoirement (FTVI, AC…) | **Dure** |

## 3. CONTRAINTES DE PRIORISATION / RISQUE

| Contrainte | Description | Type |
|---|---|---|
| `seuil_decile_prioritaire` | Déciles prioritaires (ex. ≥7) traités avant les autres | **Souple** |
| `age_max_sans_renouvellement` | Âge maximal avant obligation de renouveler (ex. 60 ans) | **Dure** |
| `fuites_detectees_seuil` | Nb fuites détectées déclenchant une urgence absolue | **Dure** |
| `quota_top10_annuel` | % minimum du top 10% risque traité chaque année | **Souple** |

## 4. CONTRAINTES RÉGLEMENTAIRES ET NORMATIVES

| Contrainte | Description | Type |
|---|---|---|
| `taux_renouvellement_min_pct` | Taux annuel imposé par le régulateur (ex. 1,5% du réseau/an) | **Dure** |
| `periodes_travaux_interdites` | Pas de travaux pendant certaines périodes (saisons, fêtes) | **Dure** |
| `conformite_qualite_eau` | Respect des normes qualité eau (OMS/nationales) | **Dure** |
| `zones_protegees` | Contraintes environnementales (nappes, zones sensibles) | **Dure** |

## 5. CONTRAINTES DE CONTINUITÉ DE SERVICE

| Contrainte | Description | Type |
|---|---|---|
| `zones_sans_coupure` | Hôpitaux, industries critiques : alimentation 24h/24 | **Dure** |
| `duree_coupure_max` | Durée max d'interruption par secteur (heures) | **Dure** |
| `redondance_reseau` | Pas de 2 chantiers simultanés sans dérivation | **Dure** |

## 6. CONTRAINTES DE COORDINATION URBAINE

| Contrainte | Description | Type |
|---|---|---|
| `coordination_voirie` | Synchroniser avec travaux de voirie | **Souple** |
| `coordination_autres_reseaux` | Coordination SOMELEC, télécoms, assainissement | **Souple** |
| `projets_urbanisme` | Éviter renouvellement dans zones en réaménagement futur | **Souple** |

---

## 7. PARAMÈTRES PAR DÉFAUT DANS LE MODÈLE

```python
CONTRAINTES_DEFAUT = {
    "budget_annuel_max":           500_000_000,   # MAD/an
    "budget_annuel_min":            50_000_000,
    "cout_km_par_materiau": {
        "FT":   8_000_000, "FTG": 7_500_000, "FTVI": 8_500_000,
        "PEHD": 6_000_000, "PVC": 5_500_000, "BTM":  7_000_000,
        "POLY": 6_500_000, "AC":  9_000_000,
    },
    "km_max_par_an":  80,
    "km_min_par_an":  10,
    "chantiers_simultanes_max": 10,
    "seuil_decile_prioritaire": 7,
    "age_max_sans_renouvellement": 60,
    "fuites_detectees_seuil": 1,
    "materiaux_urgence": ["FTVI", "AC"],
    "taux_renouvellement_min_pct": 1.5,
    "lissage_budget_pct": 0.30,
    "horizon_plan": 5,
    "annee_debut": 2026,
}
```

## 8. TYPES DE CONTRAINTES

| Type | Signification |
|---|---|
| **Dure** | Doit être respectée (violation = plan infaisable) |
| **Souple** | Relaxable avec pénalité (terme dans la fonction objectif) |
| **Paramètre** | Données d'entrée configurables |

## 9. SCÉNARIOS WHAT-IF TYPIQUES

1. **Budget +20%** → combien de km supplémentaires, quelle réduction de risque ?
2. **Traiter uniquement les fuites en urgence** → coût vs risque résiduel
3. **Priorité grands diamètres** → impact sur la continuité de service
4. **Horizon réduit de 5 à 3 ans** → intensité budgétaire et charge opérationnelle
5. **Remplacement d'urgence FTVI uniquement** → coût immédiat et bénéfice risque
