# -*- coding: utf-8 -*-
"""
Générateur de dataset synthétique d'analyse de sentiment (v2).
--------------------------------------------------------------
v2 corrige les problèmes identifiés dans une première version :
    - accords grammaticaux (genre) sur les participes passés et
      adjectifs directement attribués au sujet ("est ouvert/ouverte"),
      gérés via un mini-système de templates genrés {m|f},
    - restriction sémantique par gabarit quand un gabarit ne convient
      pas à TOUS les sujets d'un domaine (ex. "équipé d'un moteur" ne
      doit s'appliquer qu'à une voiture, pas à un garage ou une révision).

Chaque sujet est maintenant un tuple (texte, genre) et chaque gabarit
peut utiliser la syntaxe {m|f} pour les mots à accorder, ex :
    "{s} est {ouvert|ouverte} du mardi au dimanche."
-> "ce restaurant est ouvert..." / "cette brasserie est ouverte..."

Ce dataset reste SYNTHÉTIQUE (généré par combinaison), pas extrait
d'avis réels : voir le README généré en fin de script pour les
recommandations d'usage.
"""

import csv
import random
import re
from itertools import product

random.seed(42)

_GENDER_RE = re.compile(r"\{([^|{}]+)\|([^|{}]+)\}")


def render(template: str, subject_text: str, gender: str) -> str:
    """Remplace {s} par le sujet, et {masc|fem} par la forme adaptée au
    genre du sujet."""
    def _pick(match):
        masc, fem = match.group(1), match.group(2)
        return masc if gender == "m" else fem

    text = _GENDER_RE.sub(_pick, template)
    text = text.replace("{s}", subject_text)
    return text


# ==============================================================================
# 1. DOMAINES : sujets (texte, genre) + gabarits positifs/négatifs/neutres
#    Un gabarit peut être : une chaîne (s'applique à TOUS les sujets du
#    domaine), ou un tuple (chaîne, [sous-liste de sujets applicables])
#    quand il ne convient sémantiquement qu'à certains sujets.
# ==============================================================================

DOMAINS = {
    "produit_tech": {
        "subjects": [("ce smartphone", "m"), ("cet ordinateur portable", "m"), ("cette tablette", "f"),
                     ("ce casque audio", "m"), ("cette montre connectée", "f"), ("cet appareil photo", "m"),
                     ("cette enceinte bluetooth", "f"), ("ce téléviseur", "m")],
        "positive": [
            "{s} offre des performances impressionnantes pour ce prix.",
            "{s} est rapide, fluide et ne chauffe presque pas.",
            "L'autonomie de {s} dépasse largement mes attentes.",
            "{s} a un design élégant et une finition très soignée.",
            "Je suis bluffé par la qualité d'image de {s}.",
            "{s} fonctionne parfaitement depuis six mois, aucun souci.",
            "L'installation de {s} a été un jeu d'enfant.",
            "{s} vaut vraiment son prix, je suis conquis.",
            "{s} est {livré|livrée} avec tous les accessoires nécessaires, rien à ajouter.",
        ],
        "negative": [
            "{s} chauffe énormément et ralentit après quelques minutes.",
            "La batterie de {s} se vide à une vitesse alarmante.",
            "{s} a cessé de fonctionner après seulement deux semaines.",
            "Je regrette cet achat, {s} est bien moins {performant|performante} qu'annoncé.",
            "{s} présente un défaut de fabrication visible dès le déballage.",
            "Le support technique de {s} n'a jamais répondu à mes messages.",
            "{s} plante régulièrement, c'est très frustrant à l'usage.",
            "Pour ce prix, {s} est franchement {décevant|décevante} en performance.",
        ],
        "neutral": [
            "{s} est disponible en quatre coloris différents.",
            "{s} pèse environ trois cents grammes selon la fiche technique.",
            "{s} est {livré|livrée} avec un câble de charge et une notice.",
            "La garantie de {s} est valable deux ans à partir de l'achat.",
            "{s} fonctionne avec la dernière version du système d'exploitation.",
            "Le prix de {s} varie selon la capacité de stockage choisie.",
        ],
    },
    "restaurant": {
        "subjects": [("ce restaurant", "m"), ("cette brasserie", "f"), ("ce bistrot", "m"),
                     ("cette pizzeria", "f"), ("ce restaurant gastronomique", "m"), ("ce food truck", "m"),
                     ("cette table", "f")],
        "positive": [
            "{s} propose une cuisine raffinée et des produits frais.",
            "Le service à {s} était attentionné sans être envahissant.",
            "Nous avons passé un moment délicieux à {s}, tout était parfait.",
            "{s} mérite vraiment sa réputation, chaque plat était une réussite.",
            "L'accueil à {s} est chaleureux, on se sent tout de suite bien.",
            "Le rapport qualité-prix de {s} est excellent pour la région.",
            "Le chef de {s} maîtrise parfaitement les cuissons et les saveurs.",
        ],
        "negative": [
            "{s} nous a servi des plats froids et sans saveur.",
            "L'attente à {s} a duré plus d'une heure sans aucune explication.",
            "Le personnel de {s} a été désagréable du début à la fin du repas.",
            "Les prix de {s} ne sont absolument pas justifiés par la qualité.",
            "L'hygiène dans les cuisines visibles de {s} laisse à désirer.",
            "Nous ne reviendrons plus jamais à {s} après cette expérience.",
            "{s} a facturé des plats que nous n'avions même pas commandés.",
        ],
        "neutral": [
            "{s} est {ouvert|ouverte} du mardi au dimanche, midi et soir.",
            "{s} propose un menu du jour à quinze euros.",
            "La carte de {s} change selon les saisons.",
            "{s} accepte les réservations uniquement par téléphone.",
            "{s} se situe à dix minutes à pied de la gare centrale.",
            "{s} dispose d'une terrasse pouvant accueillir trente couverts.",
        ],
    },
    "hotellerie": {
        "subjects": [("cet hôtel", "m"), ("cette résidence", "f"), ("ce gîte", "m"),
                     ("cet établissement", "m"), ("cette auberge", "f"), ("ce complexe hôtelier", "m")],
        "positive": [
            "{s} offre des chambres spacieuses et impeccablement propres.",
            "Le personnel de {s} a été aux petits soins durant tout le séjour.",
            "{s} propose un petit-déjeuner copieux et varié chaque matin.",
            "La vue depuis notre chambre à {s} était absolument magnifique.",
            "{s} est {idéalement situé|idéalement située}, à deux pas de toutes les attractions.",
            "Le calme et le confort de {s} nous ont permis de vraiment nous reposer.",
        ],
        "negative": [
            "{s} était {bruyant|bruyante} toute la nuit à cause des travaux voisins.",
            "La chambre réservée à {s} sentait fortement l'humidité.",
            "{s} n'a jamais nettoyé notre chambre durant tout le séjour.",
            "La climatisation de {s} était en panne malgré nos multiples appels.",
            "Le personnel de {s} s'est montré froid et peu serviable.",
            "{s} facture des suppléments non annoncés lors de la réservation.",
        ],
        "neutral": [
            "{s} propose cent vingt chambres réparties sur cinq étages.",
            "Le check-in à {s} se fait à partir de quatorze heures.",
            "{s} dispose d'un parking privé gratuit pour les clients.",
            "{s} accepte les animaux de compagnie moyennant un supplément.",
            "Le wifi est inclus dans le tarif de la chambre à {s}.",
            "{s} se trouve à quinze minutes en voiture de l'aéroport.",
        ],
    },
    "service_client": {
        "subjects": [("le service client", "m"), ("l'assistance technique", "f"), ("le support en ligne", "m"),
                     ("l'équipe commerciale", "f"), ("le service après-vente", "m"), ("le centre d'appel", "m")],
        "positive": [
            "{s} a résolu mon problème en moins de dix minutes.",
            "{s} s'est {montré patient|montrée patiente} et {pédagogue|pédagogue} durant tout l'échange.",
            "J'ai été impressionné par la réactivité de {s}.",
            "{s} m'a rappelé comme promis et a tenu tous ses engagements.",
            "{s} a fait preuve d'une grande courtoisie malgré ma situation compliquée.",
        ],
        "negative": [
            "{s} m'a mis en attente pendant plus de quarante minutes.",
            "{s} n'a jamais répondu à mes trois emails de réclamation.",
            "{s} m'a raccroché au nez sans résoudre mon problème.",
            "{s} m'a donné des informations contradictoires à chaque appel.",
            "{s} s'est {montré condescendant|montrée condescendante} et peu à l'écoute de ma demande.",
            "J'ai dû répéter cinq fois la même chose à {s} sans résultat.",
        ],
        "neutral": [
            "{s} est joignable du lundi au vendredi de neuf à dix-huit heures.",
            "{s} propose désormais un chat en ligne en plus du téléphone.",
            "{s} traite les demandes dans un délai moyen de quarante-huit heures.",
            "{s} a mis à jour sa politique de retour le mois dernier.",
        ],
    },
    "livraison_ecommerce": {
        "subjects": [("ma commande", "f"), ("le colis", "m"), ("cette livraison", "f")],
        "positive": [
            "{s} est {arrivé|arrivée} un jour plus tôt que prévu, quelle bonne surprise.",
            "{s} était {parfaitement emballé|parfaitement emballée}, aucun dommage à l'arrivée.",
            "Le suivi de {s} était précis du début à la fin.",
            "{s} a été {livré|livrée} directement en main propre par un livreur très sympathique.",
        ],
        "negative": [
            "{s} a mis trois semaines à arriver au lieu des cinq jours annoncés.",
            "{s} est {arrivé|arrivée} complètement {écrasé|écrasée}, le contenu était inutilisable.",
            "{s} a été {livré|livrée} à la mauvaise adresse sans aucune notification.",
            "Le suivi de {s} indiquait 'livré' alors que je n'ai jamais rien reçu.",
            "{s} contenait un article différent de celui que j'avais commandé.",
        ],
        "neutral": [
            "{s} sera {livré|livrée} entre le douze et le quinze du mois.",
            "{s} est {expédié|expédiée} depuis l'entrepôt régional le plus proche.",
            "Les frais de {s} dépendent du poids total du colis.",
            "{s} peut être {suivi|suivie} en temps réel via l'application.",
        ],
    },
    "livraison_prestataires": {
        "subjects": [("ce transporteur", "m"), ("cette boutique en ligne", "f")],
        "positive": [
            "{s} respecte systématiquement les délais annoncés lors de la commande.",
            "{s} propose un service client réactif en cas de souci de livraison.",
        ],
        "negative": [
            "{s} égare régulièrement les colis qui lui sont confiés.",
            "{s} ne répond jamais aux réclamations concernant les livraisons ratées.",
        ],
        "neutral": [
            "{s} livre habituellement sous deux à cinq jours ouvrés.",
            "{s} propose plusieurs options de livraison à des tarifs variables.",
        ],
    },
    "banque_assurance": {
        "subjects": [("cette banque", "f"), ("mon assurance habitation", "f"), ("mon assurance auto", "f"),
                     ("ce conseiller financier", "m"), ("cette mutuelle", "f")],
        "positive": [
            "{s} a remboursé mon sinistre en moins d'une semaine.",
            "{s} propose des frais de gestion parmi les plus bas du marché.",
            "{s} m'a proposé un contrat parfaitement adapté à ma situation.",
            ("Le conseiller de {s} a pris le temps de bien tout m'expliquer.",
             ["cette banque", "mon assurance habitation", "mon assurance auto", "cette mutuelle"]),
            ("{s} a pris le temps de bien tout m'expliquer et de me rassurer.", ["ce conseiller financier"]),
        ],
        "negative": [
            "{s} a refusé de couvrir un sinistre pourtant prévu dans le contrat.",
            "{s} multiplie les frais cachés qui n'étaient jamais mentionnés.",
            "{s} a mis six mois à traiter mon dossier de remboursement.",
            "{s} augmente mes cotisations chaque année sans aucune justification claire.",
        ],
        "neutral": [
            "{s} propose plusieurs formules selon le niveau de couverture souhaité.",
            "Le contrat avec {s} peut être résilié à chaque date anniversaire.",
            "{s} demande un délai de carence de trois mois pour certaines garanties.",
            "Les documents pour {s} peuvent être envoyés par courrier ou en agence.",
        ],
    },
    "sante_clinique": {
        "subjects": [("cette clinique", "f"), ("ce cabinet médical", "m"), ("cet hôpital", "m"),
                     ("ce dentiste", "m"), ("ce laboratoire d'analyses", "m")],
        "positive": [
            "{s} m'a reçu rapidement malgré l'urgence de ma situation.",
            "Le personnel soignant de {s} a été d'une grande humanité.",
            "{s} dispose d'équipements modernes et d'un accueil très professionnel.",
            "Les explications données à {s} étaient claires et rassurantes.",
        ],
        "negative": [
            "{s} m'a fait attendre plus de trois heures sans aucune explication.",
            "Le diagnostic posé à {s} s'est révélé complètement erroné.",
            "{s} a perdu mon dossier médical à deux reprises.",
            "L'accueil à {s} a été particulièrement froid et expéditif.",
        ],
        "neutral": [
            "{s} est {ouvert|ouverte} du lundi au samedi sur rendez-vous.",
            "{s} accepte la carte vitale et le tiers payant.",
            "{s} se situe au deuxième étage du bâtiment médical.",
            "Les résultats d'analyses de {s} sont disponibles sous quarante-huit heures.",
        ],
    },
    "education_formation": {
        "subjects": [("cette formation", "f"), ("cette école", "f"), ("ce cours en ligne", "m"),
                     ("cet organisme de formation", "m"), ("ce programme universitaire", "m")],
        "positive": [
            "{s} est {extrêmement bien structuré|extrêmement bien structurée} et facile à suivre.",
            "Le formateur de {s} maîtrise parfaitement son sujet et transmet sa passion.",
            "{s} m'a permis d'acquérir des compétences immédiatement applicables.",
            "Le contenu de {s} est riche, actualisé et pertinent pour le marché du travail.",
        ],
        "negative": [
            "{s} est {mal organisé|mal organisée}, les supports de cours sont incomplets.",
            "Le formateur de {s} ne maîtrisait visiblement pas son sujet.",
            "{s} ne correspond absolument pas à la description du programme annoncé.",
            "J'ai trouvé {s} beaucoup trop {cher|chère} pour le contenu proposé.",
        ],
        "neutral": [
            "{s} se déroule sur douze semaines à raison de six heures par semaine.",
            "{s} délivre une attestation de fin de formation.",
            "L'inscription à {s} se fait en ligne via le site officiel.",
            "{s} est éligible au financement par le compte personnel de formation.",
        ],
    },
    "transport": {
        "subjects": [("ce vol", "m"), ("ce train", "m"), ("ce trajet en bus", "m")],
        "positive": [
            "{s} est {parti et arrivé|parti et arrivé} exactement à l'heure prévue.",
            "{s} a été très confortable malgré la longue durée du trajet.",
            "Le personnel à bord de {s} a été aux petits soins avec les passagers.",
            "{s} propose un excellent rapport qualité-prix pour cette distance.",
        ],
        "negative": [
            "{s} a été annulé à la dernière minute sans aucune compensation.",
            "{s} a subi un retard de plus de quatre heures sans explication claire.",
            "{s} était bondé et il n'y avait aucune place assise disponible.",
            "Le personnel à bord de {s} a été particulièrement désagréable avec les passagers.",
        ],
        "neutral": [
            "{s} dessert cette destination trois fois par semaine.",
            "{s} propose des billets remboursables moyennant un supplément.",
            "La durée moyenne de {s} est d'environ trois heures.",
            "{s} accepte les bagages jusqu'à vingt-trois kilogrammes en soute.",
        ],
    },
    "transport_prestataires": {
        # Sous-domaine séparé pour les sujets "personnes/entreprises" du
        # transport, dont les gabarits diffèrent sémantiquement (on ne dit
        # pas qu'une compagnie "est arrivée à l'heure", mais qu'elle "assure
        # des vols ponctuels").
        "subjects": [("ce chauffeur de taxi", "m"), ("cette compagnie aérienne", "f")],
        "positive": [
            "{s} a été très {ponctuel|ponctuelle} et {courtois|courtoise} durant tout le trajet.",
            "{s} propose un service fiable et un excellent accueil.",
        ],
        "negative": [
            "{s} a été particulièrement {impoli|impolie} avec les passagers.",
            "{s} multiplie les retards et les annulations sans explication.",
        ],
        "neutral": [
            "{s} dessert plusieurs grandes villes chaque semaine.",
            "{s} propose plusieurs formules tarifaires selon la période.",
        ],
    },
    "immobilier": {
        "subjects": [("cette agence immobilière", "f"), ("cet appartement", "m"), ("ce syndic de copropriété", "m"),
                     ("cette maison", "f"), ("ce bailleur", "m")],
        "positive": [
            "{s} a été très {réactif|réactive} et {transparent|transparente} tout au long de la transaction.",
            "{s} correspond exactement à la description et aux photos publiées.",
            "{s} a su négocier un excellent prix pour notre achat.",
            "La communication avec {s} a toujours été claire et professionnelle.",
        ],
        "negative": [
            "{s} nous a caché des vices importants découverts après l'emménagement.",
            "{s} ne répond plus depuis la signature du contrat.",
            "{s} facture des frais d'agence disproportionnés par rapport au service rendu.",
            "{s} a mis des mois à effectuer des réparations pourtant urgentes.",
        ],
        "neutral": [
            "{s} propose une surface habitable de soixante-dix mètres carrés.",
            "{s} demande un dépôt de garantie équivalent à un mois de loyer.",
            "Les charges de {s} incluent l'eau et le chauffage collectif.",
            "{s} se situe à proximité immédiate des transports en commun.",
        ],
    },
    "telecom_internet": {
        "subjects": [("mon opérateur mobile", "m"), ("ma connexion internet", "f"),
                     ("ce forfait téléphonique", "m"), ("ce fournisseur d'accès", "m")],
        "positive": [
            "{s} offre un débit stable même aux heures de forte affluence.",
            "{s} propose un excellent rapport qualité-prix comparé à la concurrence.",
            "Le passage à {s} s'est fait sans aucune coupure de service.",
            "{s} a résolu ma panne le jour même de mon signalement.",
        ],
        "negative": [
            "{s} coupe plusieurs fois par jour sans raison apparente.",
            "{s} facture des services que je n'ai jamais souscrits.",
            "{s} met des semaines à intervenir en cas de panne signalée.",
            "Le débit de {s} est très en dessous de ce qui était annoncé.",
        ],
        "neutral": [
            "{s} propose un forfait de cent gigaoctets par mois.",
            "{s} facture des frais de résiliation en cas d'engagement non terminé.",
            "L'installation de {s} nécessite l'intervention d'un technicien.",
            "{s} couvre l'ensemble du territoire national en quatrième génération.",
        ],
    },
    "sport_loisirs": {
        "subjects": [("cette salle de sport", "f"), ("ce club de fitness", "m"),
                     ("ce cours collectif", "m"), ("cette piscine municipale", "f")],
        "positive": [
            "{s} dispose d'équipements modernes et toujours bien entretenus.",
            "{s} propose des cours variés adaptés à tous les niveaux.",
            "L'ambiance à {s} est motivante et conviviale.",
            "{s} m'a permis d'atteindre mes objectifs plus vite que prévu.",
        ],
        "negative": [
            "{s} est {surpeuplé|surpeuplée} aux heures de pointe, impossible de s'entraîner correctement.",
            "Les équipements de {s} sont vétustes et souvent hors service.",
            "{s} a annulé plusieurs cours sans prévenir les inscrits.",
            "L'hygiène des vestiaires de {s} laisse vraiment à désirer.",
        ],
        "neutral": [
            "{s} est {ouvert|ouverte} de six heures à vingt-trois heures en semaine.",
            "{s} propose un abonnement mensuel sans engagement.",
            "{s} dispose d'un parking gratuit réservé aux adhérents.",
            "L'accès à {s} nécessite une carte magnétique personnelle.",
        ],
    },
    "sport_coach": {
        "subjects": [("ce coach sportif", "m")],
        "positive": [
            "{s} est très pédagogue et sait adapter les séances à mon niveau.",
            "{s} m'a permis de progresser rapidement grâce à des conseils personnalisés.",
        ],
        "negative": [
            "{s} n'a jamais adapté les séances à ma condition physique.",
            "{s} arrive systématiquement en retard aux rendez-vous fixés.",
        ],
        "neutral": [
            "{s} propose des séances individuelles ou en petit groupe.",
            "{s} intervient dans plusieurs salles de la région.",
        ],
    },
    "automobile": {
        "subjects": [("ce garage", "m"), ("ce concessionnaire", "m"), ("ce loueur de véhicules", "m")],
        "positive": [
            "{s} a fait un diagnostic précis et une réparation impeccable.",
            "{s} propose des tarifs transparents et sans surprise.",
            "Le personnel de {s} a été honnête et n'a facturé que le nécessaire.",
        ],
        "negative": [
            "{s} a facturé des réparations qui n'étaient absolument pas nécessaires.",
            "{s} a rendu le véhicule avec une nouvelle panne non signalée avant.",
            "{s} nous a fait attendre des semaines pour un simple contrôle.",
            "Le devis de {s} a doublé sans explication le jour de la restitution.",
        ],
        "neutral": [
            "{s} propose une révision tous les vingt mille kilomètres.",
            "{s} accepte les paiements en plusieurs fois sans frais.",
        ],
    },
    "automobile_vehicule": {
        "subjects": [("cette voiture", "f"), ("cette révision", "f")],
        "positive": [
            ("{s} consomme peu et reste très agréable à conduire au quotidien.", ["cette voiture"]),
            ("{s} a été effectuée rapidement et sans surcoût par rapport au devis initial.", ["cette révision"]),
        ],
        "negative": [
            ("{s} tombe en panne bien plus souvent qu'attendu pour son âge.", ["cette voiture"]),
            ("{s} n'a pas résolu le problème signalé, la panne est réapparue une semaine après.", ["cette révision"]),
        ],
        "neutral": [
            ("{s} est {équipé|équipée} d'un moteur essence de cent chevaux.", ["cette voiture"]),
            "La garantie de {s} couvre deux ans ou cent mille kilomètres.",
        ],
    },
    "beaute_cosmetique": {
        "subjects": [("ce salon de coiffure", "m"), ("cet institut de beauté", "m"), ("ce spa", "m")],
        "positive": [
            "{s} a fait un travail remarquable, exactement ce que je voulais.",
            "{s} utilise des produits de qualité et prend le temps nécessaire.",
            "Le résultat obtenu chez {s} a dépassé toutes mes attentes.",
            "L'ambiance chez {s} est apaisante et le personnel très attentionné.",
        ],
        "negative": [
            "{s} n'a pas du tout respecté ce que j'avais demandé.",
            "Le résultat chez {s} était très éloigné de ce qui était promis.",
            "{s} a facturé bien plus cher que le tarif annoncé initialement.",
        ],
        "neutral": [
            "{s} propose des prestations sur rendez-vous uniquement.",
            "{s} est ouvert du mardi au samedi.",
            "Les tarifs de {s} varient selon la prestation choisie.",
            "{s} accepte les cartes cadeaux et les paiements en ligne.",
        ],
    },
    "beaute_produit": {
        "subjects": [("cette crème visage", "f")],
        "positive": [
            "{s} a nettement amélioré l'apparence de ma peau en quelques semaines.",
            "{s} a une texture agréable et une odeur discrète.",
        ],
        "negative": [
            "{s} a irrité ma peau dès les premières utilisations.",
            "{s} ne tient absolument pas les promesses affichées sur l'emballage.",
        ],
        "neutral": [
            "{s} s'applique matin et soir sur peau propre.",
            "{s} est disponible en format voyage ou en format standard.",
        ],
    },
    "administration_publique": {
        "subjects": [("cette mairie", "f"), ("cette préfecture", "f"),
                     ("ce service des impôts", "m"), ("cet organisme administratif", "m")],
        "positive": [
            "{s} a traité mon dossier rapidement et sans complication.",
            "L'agent reçu à {s} a été de bon conseil et très patient.",
            "{s} propose désormais des démarches en ligne très pratiques.",
            "{s} m'a rappelé pour m'informer de l'avancement de ma demande.",
        ],
        "negative": [
            "{s} m'a fait attendre des mois sans aucune nouvelle de mon dossier.",
            "{s} a égaré des documents pourtant déposés en main propre.",
            "L'accueil à {s} a été particulièrement froid et peu aidant.",
            "{s} exige des documents contradictoires selon les guichets.",
        ],
        "neutral": [
            "{s} est {ouvert|ouverte} du lundi au vendredi sur rendez-vous.",
            "{s} propose un service en ligne pour certaines démarches.",
            "Les délais de traitement à {s} varient selon la période de l'année.",
            "{s} demande une pièce d'identité valide pour tout dépôt de dossier.",
        ],
    },
    "culture_spectacle": {
        "subjects": [("ce concert", "m"), ("cette exposition", "f"), ("ce spectacle", "m"),
                     ("ce festival", "m"), ("ce musée", "m")],
        "positive": [
            "{s} était une expérience inoubliable du début à la fin.",
            "{s} a proposé une programmation riche et originale.",
            "L'organisation de {s} était irréprochable malgré l'affluence.",
            "{s} vaut vraiment le détour, je le recommande sans hésiter.",
        ],
        "negative": [
            "{s} a été annulé au dernier moment sans aucun remboursement.",
            "{s} était bien trop {cher|chère} pour ce qui était réellement proposé.",
            "L'organisation de {s} était chaotique, files d'attente interminables.",
            "{s} n'a pas du tout tenu ses promesses par rapport à la communication.",
        ],
        "neutral": [
            "{s} se tient chaque année à la même période.",
            "Les billets pour {s} sont disponibles en ligne ou sur place.",
            "{s} accueille en moyenne plusieurs milliers de visiteurs par jour.",
            "{s} dure environ deux heures, entracte compris.",
        ],
    },
    "emploi_rh": {
        "subjects": [("cette entreprise", "f"), ("cet employeur", "m")],
        "positive": [
            "{s} propose un environnement de travail vraiment bienveillant.",
            "{s} a été {transparent|transparente} sur les conditions dès le premier entretien.",
            "{s} valorise réellement l'équilibre entre vie professionnelle et personnelle.",
            "L'intégration chez {s} a été particulièrement bien organisée.",
        ],
        "negative": [
            "{s} impose une charge de travail largement excessive sans reconnaissance.",
            "{s} a annoncé un salaire différent de celui négocié à l'embauche.",
            "L'ambiance chez {s} est délétère et le turnover très élevé.",
        ],
        "neutral": [
            "{s} compte environ deux cents salariés répartis sur trois sites.",
            "{s} propose un poste en contrat à durée indéterminée.",
            "{s} est {basé|basée} dans la zone industrielle en périphérie de la ville.",
        ],
    },
    "emploi_recrutement": {
        "subjects": [("ce processus de recrutement", "m")],
        "positive": [
            "{s} a été transparent et bien organisé du début à la fin.",
            "{s} comportait des entretiens pertinents et bien préparés.",
        ],
        "negative": [
            "{s} n'a jamais donné suite après trois entretiens successifs.",
            "{s} a traîné en longueur sans aucune communication claire.",
        ],
        "neutral": [
            "{s} comporte généralement trois entretiens successifs.",
            "{s} se déroule intégralement en visioconférence.",
        ],
    },
    "logiciel_application": {
        "subjects": [("cette application", "f"), ("ce logiciel", "m"),
                     ("cette plateforme en ligne", "f"), ("ce site web", "m")],
        "positive": [
            "{s} est {intuitif|intuitive} et agréable à utiliser au quotidien.",
            "{s} a considérablement simplifié ma gestion administrative.",
            "Les mises à jour de {s} apportent régulièrement de vraies améliorations.",
            "{s} fonctionne parfaitement sans aucun bug depuis l'installation.",
        ],
        "negative": [
            "{s} plante constamment et fait perdre un temps précieux.",
            "{s} est {devenu payant|devenue payante} du jour au lendemain sans prévenir les utilisateurs.",
            "L'interface de {s} est confuse et peu intuitive.",
            "{s} a perdu toutes mes données lors de la dernière mise à jour.",
        ],
        "neutral": [
            "{s} est disponible sur ordinateur, tablette et smartphone.",
            "{s} nécessite la création d'un compte pour être utilisée.",
            "{s} propose une version gratuite avec des fonctionnalités limitées.",
            "{s} est {mis à jour|mise à jour} environ une fois par mois.",
        ],
    },
    "alimentation_supermarche": {
        "subjects": [("ce supermarché", "m"), ("cette épicerie", "f")],
        "positive": [
            "{s} propose des produits frais et de très bonne qualité.",
            "{s} a des prix très compétitifs pour des produits locaux.",
            "Le personnel de {s} est toujours souriant et disponible.",
            "{s} a un large choix de produits, même pour les régimes spécifiques.",
        ],
        "negative": [
            "{s} vend régulièrement des produits proches de la date de péremption.",
            "Les rayons de {s} sont souvent vides pour les produits de base.",
            "{s} a doublé ses prix sans aucune amélioration de la qualité.",
            "L'hygiène de {s} laisse clairement à désirer.",
        ],
        "neutral": [
            "{s} est {ouvert|ouverte} tous les jours de huit à vingt heures.",
            "{s} propose un service de livraison à domicile.",
            "{s} accepte les tickets restaurant.",
            "{s} dispose d'un rayon dédié aux produits sans gluten.",
        ],
    },
    "alimentation_produit": {
        "subjects": [("ce produit alimentaire", "m"), ("cette marque de produits bio", "f")],
        "positive": [
            "{s} a un excellent goût et une composition irréprochable.",
            "{s} respecte vraiment ses engagements en matière de qualité biologique.",
        ],
        "negative": [
            "{s} a un goût très éloigné de ce que la publicité laissait penser.",
            "{s} contient des additifs qui ne sont pas clairement indiqués sur l'emballage.",
        ],
        "neutral": [
            "{s} est disponible en plusieurs formats selon les points de vente.",
            "{s} affiche une liste d'ingrédients complète sur l'emballage.",
        ],
    },
    "voyage_tourisme": {
        "subjects": [("cette agence de voyage", "f"), ("ce circuit touristique", "m"),
                     ("cette croisière", "f"), ("ce séjour organisé", "m")],
        "positive": [
            ("{s} a été {parfaitement organisé|parfaitement organisée} du début à la fin.",
             ["cette agence de voyage", "ce circuit touristique", "cette croisière"]),
            ("{s} s'est déroulé sans le moindre accroc, une réussite totale.", ["ce séjour organisé"]),
            "{s} nous a fait découvrir des lieux magnifiques hors des sentiers battus.",
            "Le guide de {s} était passionnant et très compétent.",
            "{s} propose un excellent rapport qualité-prix pour ce type de voyage.",
        ],
        "negative": [
            "{s} a été bien en dessous de ce qui était décrit dans la brochure.",
            "{s} a annulé plusieurs excursions sans aucun remboursement.",
            "Le logement inclus dans {s} était très éloigné de la description.",
            "{s} a facturé des suppléments non annoncés à chaque étape.",
        ],
        "neutral": [
            "{s} dure en moyenne huit jours et sept nuits.",
            "{s} inclut les vols, l'hébergement et certains repas.",
            "{s} est {proposé|proposée} à différentes dates entre avril et octobre.",
            "{s} nécessite un acompte de trente pour cent à la réservation.",
        ],
    },
}

# ==============================================================================
# 2. CONNECTEURS ET VARIATIONS SYNTAXIQUES
# ==============================================================================

POSITIVE_INTROS = ["", "Honnêtement, ", "Franchement, ", "Sans hésiter, ", "Je dois dire que ", "Sincèrement, "]
NEGATIVE_INTROS = ["", "Malheureusement, ", "Honnêtement, ", "Sans surprise, ", "Je regrette de dire que ", "Pour être franc, "]
NEUTRAL_INTROS = ["", "Pour information, ", "À noter que ", "Selon le site officiel, "]

POSITIVE_SUFFIXES = ["", " Je recommande vivement.", " Une très belle surprise.", " Rien à redire.", " Bravo à toute l'équipe."]
NEGATIVE_SUFFIXES = ["", " Je ne recommande pas du tout.", " Quelle déception.", " À éviter absolument.", " Plus jamais."]
NEUTRAL_SUFFIXES = ["", " Plus d'informations sur le site officiel.", " Voir les conditions générales."]


def _lowercase_first(sentence: str) -> str:
    if not sentence:
        return sentence
    return sentence[0].lower() + sentence[1:]


def _applicable_subjects(entry, all_subjects: list[tuple]) -> list[tuple]:
    """Un gabarit peut être une simple chaîne (s'applique à tous les
    sujets du domaine) ou un tuple (chaîne, [noms de sujets autorisés])
    pour restreindre son usage à des sujets sémantiquement cohérents."""
    if isinstance(entry, tuple):
        template, allowed_names = entry
        return template, [pair for pair in all_subjects if pair[0] in allowed_names]
    return entry, all_subjects


def generate_rows(max_per_domain_per_class: int = 60) -> list[tuple]:
    rows = set()

    for domain, spec in DOMAINS.items():
        all_subjects = spec["subjects"]

        for label, templates, intros, suffixes in [
            ("positif", spec["positive"], POSITIVE_INTROS, POSITIVE_SUFFIXES),
            ("négatif", spec["negative"], NEGATIVE_INTROS, NEGATIVE_SUFFIXES),
            ("neutre", spec["neutral"], NEUTRAL_INTROS, NEUTRAL_SUFFIXES),
        ]:
            candidates = []
            for entry in templates:
                template, subjects_for_template = _applicable_subjects(entry, all_subjects)
                for (subject_text, gender), intro, suffix in product(subjects_for_template, intros, suffixes):
                    candidates.append((template, subject_text, gender, intro, suffix))

            random.shuffle(candidates)
            count = 0
            for template, subject_text, gender, intro, suffix in candidates:
                if count >= max_per_domain_per_class:
                    break
                base = render(template, subject_text, gender)
                if intro:
                    base = intro + _lowercase_first(base)
                else:
                    base = base[0].upper() + base[1:] if base else base
                sentence = " ".join((base + suffix).split())
                if (sentence, label) not in rows:
                    rows.add((sentence, label))
                    count += 1

    return list(rows)


if __name__ == "__main__":
    rows = generate_rows(max_per_domain_per_class=60)
    random.shuffle(rows)

    from collections import Counter
    counts = Counter(label for _, label in rows)
    print(f"Total généré : {len(rows)} phrases")
    print("Répartition par classe :", dict(counts))

    with open("/home/claude/dataset_generation/dataset_synthetique.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print("\nExemples aléatoires :")
    for text, label in rows[:15]:
        print(f"  [{label:<8}] {text}")
