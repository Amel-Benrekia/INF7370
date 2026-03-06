# ==========================================================
# Tâche 2 : Extraction des caractéristiques
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# 0. Chargement des données (avec reconstruction des metadata)
# ----------------------------------------------------------

# ===== 1. Profils des pollueurs =====
polluters_raw = pd.read_csv(
    "data/content_polluters.txt",
    sep=r"\s+",
    header=None,
    engine="python"
)

polluters_users = pd.DataFrame({
    "UserID": polluters_raw[0],
    "CreatedAt": polluters_raw[1] + " " + polluters_raw[2],
    "CollectedAt": polluters_raw[3] + " " + polluters_raw[4],
    "NumberOfFollowings": polluters_raw[5],
    "NumberOfFollowers": polluters_raw[6],
    "NumberOfTweets": polluters_raw[7],
    "LengthOfScreenName": polluters_raw[8],
    "LengthOfDescriptionInUserProfile": polluters_raw[9]
})

# ===== 2. Profils des utilisateurs légitimes =====
legit_raw = pd.read_csv(
    "data/legitimate_users.txt",
    sep=r"\s+",
    header=None,
    engine="python"
)

legitimate_users = pd.DataFrame({
    "UserID": legit_raw[0],
    "CreatedAt": legit_raw[1] + " " + legit_raw[2],
    "CollectedAt": legit_raw[3] + " " + legit_raw[4],
    "NumberOfFollowings": legit_raw[5],
    "NumberOfFollowers": legit_raw[6],
    "NumberOfTweets": legit_raw[7],
    "LengthOfScreenName": legit_raw[8],
    "LengthOfDescriptionInUserProfile": legit_raw[9]
})

# ===== 3. Tweets des pollueurs =====
polluters_tweets = pd.read_csv(
    "data/content_polluters_tweets.txt",
    sep="\t",
    header=None,
    names=["UserID", "TweetID", "Tweet", "CreatedAt"]
)

# ===== 4. Tweets des utilisateurs légitimes =====
legitimate_tweets = pd.read_csv(
    "data/legitimate_users_tweets.txt",
    sep="\t",
    header=None,
    names=["UserID", "TweetID", "Tweet", "CreatedAt"]
)

# ===== 5. Séries de followings des pollueurs =====
polluters_followings = pd.read_csv(
    "data/content_polluters_followings.txt",
    sep="\t",
    header=None,
    names=["UserID", "SeriesOfNumberOfFollowings"]
)

# ===== 6. Séries de followings des utilisateurs légitimes =====
legitimate_followings = pd.read_csv(
    "data/legitimate_users_followings.txt",
    sep="\t",
    header=None,
    names=["UserID", "SeriesOfNumberOfFollowings"]
)

# ----------------------------------------------------------
# Vérification rapide
# ----------------------------------------------------------

print("Polluters users:", polluters_users.shape)
print("Legitimate users:", legitimate_users.shape)

print("Polluters tweets:", polluters_tweets.shape)
print("Legitimate tweets:", legitimate_tweets.shape)

print("Polluters followings:", polluters_followings.shape)
print("Legitimate followings:", legitimate_followings.shape)

# ----------------------------------------------------------
# 1. Longueur du nom d'utilisateur
# ----------------------------------------------------------

# Cette caractéristique est déjà fournie dans le dataset
# sous le nom "LengthOfScreenName".
# Nous la conservons telle quelle.

polluters_users["username_length"] = polluters_users["LengthOfScreenName"]
legitimate_users["username_length"] = legitimate_users["LengthOfScreenName"]

# ----------------------------------------------------------
# 2. Longueur de la description du profil
# ----------------------------------------------------------

# Caractéristique fournie dans le dataset
# Nous la renommons pour plus de clarté

polluters_users["description_length"] = polluters_users["LengthOfDescriptionInUserProfile"]
legitimate_users["description_length"] = legitimate_users["LengthOfDescriptionInUserProfile"]

# ----------------------------------------------------------
# 3. Durée de vie du compte (en jours)
# ----------------------------------------------------------

# Conversion des colonnes de dates en format datetime
polluters_users["CreatedAt"] = pd.to_datetime(polluters_users["CreatedAt"])
polluters_users["CollectedAt"] = pd.to_datetime(polluters_users["CollectedAt"])

legitimate_users["CreatedAt"] = pd.to_datetime(legitimate_users["CreatedAt"])
legitimate_users["CollectedAt"] = pd.to_datetime(legitimate_users["CollectedAt"])

# Calcul de la durée de vie en jours
polluters_users["account_age_days"] = (
    polluters_users["CollectedAt"] - polluters_users["CreatedAt"]
).dt.days

legitimate_users["account_age_days"] = (
    legitimate_users["CollectedAt"] - legitimate_users["CreatedAt"]
).dt.days

# Si jamais il existe des valeurs négatives ou nulles,
# on les remplace par 0 par sécurité.
polluters_users["account_age_days"] = polluters_users["account_age_days"].clip(lower=0)
legitimate_users["account_age_days"] = legitimate_users["account_age_days"].clip(lower=0)

# ----------------------------------------------------------
# 4. Nombre de followings
# ----------------------------------------------------------

# Caractéristique directement fournie dans le dataset
polluters_users["num_followings"] = polluters_users["NumberOfFollowings"]
legitimate_users["num_followings"] = legitimate_users["NumberOfFollowings"]

# ----------------------------------------------------------
# 5. Nombre de followers
# ----------------------------------------------------------

# Caractéristique directement fournie dans le dataset
polluters_users["num_followers"] = polluters_users["NumberOfFollowers"]
legitimate_users["num_followers"] = legitimate_users["NumberOfFollowers"]

# ----------------------------------------------------------
# 6. Ratio following / followers
# ----------------------------------------------------------

def compute_follow_ratio(row):
    """
    Calcule le ratio entre le nombre de followings
    et le nombre de followers.

    Si le nombre de followers est égal à 0,
    le ratio est fixé à 0 pour éviter une division par zéro.
    """
    if row["NumberOfFollowers"] == 0:
        return 0
    return row["NumberOfFollowings"] / row["NumberOfFollowers"]


# Application de la fonction sur les deux datasets
polluters_users["follow_ratio"] = polluters_users.apply(compute_follow_ratio, axis=1)
legitimate_users["follow_ratio"] = legitimate_users.apply(compute_follow_ratio, axis=1)

# ----------------------------------------------------------
# 7. Nombre moyen de tweets par jour
# ----------------------------------------------------------

def compute_tweets_per_day(row):
    """
    Calcule le nombre moyen de tweets par jour.

    Si la durée de vie du compte est nulle,
    la valeur retournée est 0.
    """
    if row["account_age_days"] == 0:
        return 0
    return row["NumberOfTweets"] / row["account_age_days"]


polluters_users["tweets_per_day"] = polluters_users.apply(compute_tweets_per_day, axis=1)
legitimate_users["tweets_per_day"] = legitimate_users.apply(compute_tweets_per_day, axis=1)

# ----------------------------------------------------------
# 8. Proportion d'URL dans les tweets
# ----------------------------------------------------------

def compute_url_proportion(tweets_df):
    """
    Calcule la proportion de tweets contenant une URL
    pour chaque utilisateur.

    Une URL est détectée par la présence de la chaîne 'http'.
    """

    # Regroupement des tweets par utilisateur
    grouped = tweets_df.groupby("UserID")

    # Calcul de la proportion
    url_proportion = grouped["Tweet"].apply(
        lambda tweets: tweets.str.contains("http", case=False, na=False).mean()
    )

    return url_proportion

# Calcul pour pollueurs
polluters_url_prop = compute_url_proportion(polluters_tweets)

# Calcul pour utilisateurs légitimes
legitimate_url_prop = compute_url_proportion(legitimate_tweets)

# Fusion avec les profils
polluters_users = polluters_users.merge(
    polluters_url_prop.rename("url_proportion"),
    on="UserID",
    how="left"
)

legitimate_users = legitimate_users.merge(
    legitimate_url_prop.rename("url_proportion"),
    on="UserID",
    how="left"
)

# ----------------------------------------------------------
# 9. Proportion de mentions (@) dans les tweets
# ----------------------------------------------------------

def compute_mention_proportion(tweets_df):
    """
    Calcule la proportion de tweets contenant
    une mention '@' pour chaque utilisateur.
    """

    grouped = tweets_df.groupby("UserID")

    mention_proportion = grouped["Tweet"].apply(
        lambda tweets: tweets.str.contains("@", na=False).mean()
    )

    return mention_proportion

polluters_mention_prop = compute_mention_proportion(polluters_tweets)
legitimate_mention_prop = compute_mention_proportion(legitimate_tweets)

polluters_users = polluters_users.merge(
    polluters_mention_prop.rename("mention_proportion"),
    on="UserID",
    how="left"
)

legitimate_users = legitimate_users.merge(
    legitimate_mention_prop.rename("mention_proportion"),
    on="UserID",
    how="left"
)

# ----------------------------------------------------------
# 10. Temps moyen et maximal entre tweets consécutifs
# ----------------------------------------------------------

def compute_time_between_tweets(tweets_df):
    """
    Calcule le temps moyen et maximal (en secondes)
    entre deux tweets consécutifs pour chaque utilisateur.

    Étapes :
    1. Regrouper les tweets par UserID
    2. Trier les tweets par date
    3. Calculer les différences temporelles
    4. Retourner la moyenne et le maximum
    """

    # Conversion des dates en format datetime
    tweets_df["CreatedAt"] = pd.to_datetime(tweets_df["CreatedAt"])

    def time_features(group):
        # Trier les tweets par date
        sorted_times = group.sort_values("CreatedAt")["CreatedAt"]

        # Calcul des différences entre tweets consécutifs
        diffs = sorted_times.diff().dt.total_seconds().dropna()

        # Si un utilisateur a moins de 2 tweets
        if len(diffs) == 0:
            return pd.Series([0, 0])

        return pd.Series([diffs.mean(), diffs.max()])

    # Application par utilisateur
    time_stats = tweets_df.groupby("UserID").apply(time_features)

    # Renommer les colonnes
    time_stats.columns = ["avg_time_between_tweets", "max_time_between_tweets"]

    return time_stats

# Calcul pour pollueurs
polluters_time_features = compute_time_between_tweets(polluters_tweets)

# Calcul pour utilisateurs légitimes
legitimate_time_features = compute_time_between_tweets(legitimate_tweets)

# Fusion avec profils
polluters_users = polluters_users.merge(
    polluters_time_features,
    on="UserID",
    how="left"
)

legitimate_users = legitimate_users.merge(
    legitimate_time_features,
    on="UserID",
    how="left"
)

# ----------------------------------------------------------
# 11. Proportion de hashtags (#) dans les tweets
# ----------------------------------------------------------

def compute_hashtag_proportion(tweets_df):
    """
    Calcule la proportion de tweets contenant un hashtag (#)
    pour chaque utilisateur.
    """

    grouped = tweets_df.groupby("UserID")

    hashtag_proportion = grouped["Tweet"].apply(
        lambda tweets: tweets.str.contains("#", na=False).mean()
    )

    return hashtag_proportion

# Calcul pour pollueurs
polluters_hashtag_prop = compute_hashtag_proportion(polluters_tweets)

# Calcul pour utilisateurs légitimes
legitimate_hashtag_prop = compute_hashtag_proportion(legitimate_tweets)

# Fusion avec profils
polluters_users = polluters_users.merge(
    polluters_hashtag_prop.rename("hashtag_proportion"),
    on="UserID",
    how="left"
)

legitimate_users = legitimate_users.merge(
    legitimate_hashtag_prop.rename("hashtag_proportion"),
    on="UserID",
    how="left"
)

# ----------------------------------------------------------
# 12. Variance du nombre de followings
# ----------------------------------------------------------

def compute_following_variance(followings_df):
    """
    Calcule la variance du nombre de comptes suivis (followings)
    pour chaque utilisateur à partir de la série historique.
    """

    variances = []

    # Parcourir chaque utilisateur
    for _, row in followings_df.iterrows():

        # Récupérer la série sous forme de texte
        series = str(row["SeriesOfNumberOfFollowings"])

        # Séparer les valeurs avec la virgule
        numbers = series.split(",")

        # Convertir les valeurs en nombres
        numbers = [int(x) for x in numbers if x != ""]

        # Calculer la variance
        if len(numbers) > 1:
            var = np.var(numbers)
        else:
            var = 0

        variances.append(var)

    # Créer un dataframe résultat
    result = pd.DataFrame({
        "UserID": followings_df["UserID"],
        "following_variance": variances
    })

    return result


# Calcul pour les pollueurs
polluters_following_var = compute_following_variance(polluters_followings)

# Calcul pour les utilisateurs légitimes
legitimate_following_var = compute_following_variance(legitimate_followings)


# Fusion avec les profils utilisateurs
polluters_users = polluters_users.merge(
    polluters_following_var,
    on="UserID",
    how="left"
)

legitimate_users = legitimate_users.merge(
    legitimate_following_var,
    on="UserID",
    how="left"
)