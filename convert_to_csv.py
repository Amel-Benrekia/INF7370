import pandas as pd

# ==========================================================
# 1. CHARGER ET CORRIGER content_polluters.txt
# ==========================================================

# Le fichier est séparé par des espaces
# Les dates sont coupées en deux colonnes (date + heure)
polluters = pd.read_csv(
    "data\content_polluters.txt",
    sep=r"\s+",
    header=None
)

# Reconstruction des colonnes de date
polluters["CreatedAt"] = polluters[1] + " " + polluters[2]
polluters["CollectedAt"] = polluters[3] + " " + polluters[4]

# Création du dataframe avec les bonnes colonnes
polluters_clean = pd.DataFrame({
    "UserID": polluters[0],
    "CreatedAt": polluters["CreatedAt"],
    "CollectedAt": polluters["CollectedAt"],
    "NumberOfFollowings": polluters[5],
    "NumberOfFollowers": polluters[6],
    "NumberOfTweets": polluters[7],
    "LengthOfScreenName": polluters[8],
    "LengthOfDescriptionInUserProfile": polluters[9]
})

# Sauvegarde en CSV
polluters_clean.to_csv("data/content_polluters.csv", index=False)


# ==========================================================
# 2. CHARGER ET CORRIGER legitimate_users.txt
# ==========================================================

legit = pd.read_csv(
    "data/legitimate_users.txt",
    sep=r"\s+",
    header=None
)

legit["CreatedAt"] = legit[1] + " " + legit[2]
legit["CollectedAt"] = legit[3] + " " + legit[4]

legit_clean = pd.DataFrame({
    "UserID": legit[0],
    "CreatedAt": legit["CreatedAt"],
    "CollectedAt": legit["CollectedAt"],
    "NumberOfFollowings": legit[5],
    "NumberOfFollowers": legit[6],
    "NumberOfTweets": legit[7],
    "LengthOfScreenName": legit[8],
    "LengthOfDescriptionInUserProfile": legit[9]
})

legit_clean.to_csv("data/legitimate_users.csv", index=False)


# ==========================================================
# 3. CHARGER content_polluters_followings.txt
# ==========================================================

polluters_followings = pd.read_csv(
    "data/content_polluters_followings.txt",
    sep="\t",
    header=None,
    names=["UserID", "SeriesOfNumberOfFollowings"]
)

polluters_followings.to_csv("data/content_polluters_followings.csv", index=False)


# ==========================================================
# 4. CHARGER legitimate_users_followings.txt
# ==========================================================

legit_followings = pd.read_csv(
    "data/legitimate_users_followings.txt",
    sep="\t",
    header=None,
    names=["UserID", "SeriesOfNumberOfFollowings"]
)

legit_followings.to_csv("data/legitimate_users_followings.csv", index=False)


# ==========================================================
# 5. CHARGER content_polluters_tweets.txt
# ==========================================================

polluters_tweets = pd.read_csv(
    "data/content_polluters_tweets.txt",
    sep="\t",
    header=None,
    names=["UserID", "TweetID", "Tweet", "CreatedAt"]
)

polluters_tweets.to_csv("data/content_polluters_tweets.csv", index=False)


# ==========================================================
# 6. CHARGER legitimate_users_tweets.txt
# ==========================================================

legit_tweets = pd.read_csv(
    "data/legitimate_users_tweets.txt",
    sep="\t",
    header=None,
    names=["UserID", "TweetID", "Tweet", "CreatedAt"]
)

legit_tweets.to_csv("data/legitimate_users_tweets.csv", index=False)

print("Tous les fichiers ont été convertis en CSV avec leurs métadonnées.")