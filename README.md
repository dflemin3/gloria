<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

## Hierarchical Bayesian analysis of the 2019-2020 NHL season (until it got canceled)

---

I miss hockey, so I decided to build a simple hierarchical Bayesian model for the 2019-2020 NHL season. Since this model is fully probabilistic, I will be able to assess the offensive and defensive ability of teams. Furthermore, I can draw samples from the posterior distribution to simulate entire seasons, and the post-season, to see who is the true Cup champ.

This work is based in large part on [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf), [Daniel Weitzenfeld's great write-up](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/), and a [nice application of this type of modeling to Rugby from the pymc3 docs](https://docs.pymc.io/notebooks/rugby_analytics.html).


```python
%matplotlib inline

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import corner
from datetime import datetime

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
```

### Get the data

---

I will use pandas to scrape the 2019-2020 NHL season schedule and results from  [www.hockey-reference.com](https://www.hockey-reference.com/leagues/NHL_2020_games.html). Note that I will only get the results of every game played up to COVID-19 season suspension, but I will also have the schedule so I can simulate the rest of the season as well.


```python
url = "http://www.hockey-reference.com/leagues/NHL_2020_games.html"
df = pd.read_html(url, parse_dates=True, attrs = {'id': 'games'},
                  header=0, index_col=0)[0]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Visitor</th>
      <th>G</th>
      <th>Home</th>
      <th>G.1</th>
      <th>Unnamed: 5</th>
      <th>Att.</th>
      <th>LOG</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-02</th>
      <td>Vancouver Canucks</td>
      <td>2.0</td>
      <td>Edmonton Oilers</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>18347.0</td>
      <td>2:23</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>Washington Capitals</td>
      <td>3.0</td>
      <td>St. Louis Blues</td>
      <td>2.0</td>
      <td>OT</td>
      <td>18096.0</td>
      <td>2:32</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>Ottawa Senators</td>
      <td>3.0</td>
      <td>Toronto Maple Leafs</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>19612.0</td>
      <td>2:36</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>San Jose Sharks</td>
      <td>1.0</td>
      <td>Vegas Golden Knights</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>18588.0</td>
      <td>2:44</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>Arizona Coyotes</td>
      <td>1.0</td>
      <td>Anaheim Ducks</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>17174.0</td>
      <td>2:25</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Looks like I have some data cleaning to do. For this modeling, we do not care about the attendance, length of the game (LOG), nor the Notes column, which is just empty. We can drop those.


```python
df.drop(columns=["Att.", "LOG", "Notes"], inplace=True)
```

I also want to rename the columns to be a bit more informative.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Visitor</th>
      <th>G</th>
      <th>Home</th>
      <th>G.1</th>
      <th>Unnamed: 5</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-02</th>
      <td>Vancouver Canucks</td>
      <td>2.0</td>
      <td>Edmonton Oilers</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>Washington Capitals</td>
      <td>3.0</td>
      <td>St. Louis Blues</td>
      <td>2.0</td>
      <td>OT</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>Ottawa Senators</td>
      <td>3.0</td>
      <td>Toronto Maple Leafs</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>San Jose Sharks</td>
      <td>1.0</td>
      <td>Vegas Golden Knights</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>Arizona Coyotes</td>
      <td>1.0</td>
      <td>Anaheim Ducks</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ["awayTeam", "awayGoals", "homeTeam", "homeGoals", "Extra"]

# Fill in NaN with Reg for Regulation in the column that indicates whether
# or not a game went into OT/SO
df["Extra"].fillna("Reg", inplace=True)
```

Before I drop it, I want to use the Extra column to estimate the empirical probability that a game ends in a shootout, given OT. I will use this value later to help simulate games. I will not calculate this on a team-by-team basis just to keep it simple, but that change could improve the model in future iterations.


```python
counts = df["Extra"].value_counts()
probSO = counts["SO"]/(counts["OT"] + counts["SO"])
print("Empirical probability of a team going to a SO, given OT: %lf" % probSO)
```

    Empirical probability of a team going to a SO, given OT: 0.344000



```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>awayTeam</th>
      <th>awayGoals</th>
      <th>homeTeam</th>
      <th>homeGoals</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-04</th>
      <td>Chicago Blackhawks</td>
      <td>NaN</td>
      <td>New York Rangers</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>Pittsburgh Penguins</td>
      <td>NaN</td>
      <td>Ottawa Senators</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>Anaheim Ducks</td>
      <td>NaN</td>
      <td>San Jose Sharks</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>Montreal Canadiens</td>
      <td>NaN</td>
      <td>Toronto Maple Leafs</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>Vegas Golden Knights</td>
      <td>NaN</td>
      <td>Vancouver Canucks</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
  </tbody>
</table>
</div>



Now let's transforms team names into their initials, e.g. transform St. Louis Blues to STL, as that will make things a bit easier down the road. Then, I want to add columns to indicate who is the away team, who is the home team, and a unique id for each team for bookkeeping.

To map the names to their initials, I will make a simple mapping dictionary to apply to the Visitor and Home columns.


```python
# Dictionary of NHL team names and abbreviations
conv = {'Anaheim Ducks' : 'ANA',
        'Arizona Coyotes' : 'ARI',
        'Boston Bruins' : 'BOS',
        'Buffalo Sabres' : 'BUF',
        'Calgary Flames' : 'CGY',
        'Carolina Hurricanes' : 'CAR',
        'Chicago Blackhawks' : 'CHI',
        'Colorado Avalanche' : 'COL',
        'Columbus Blue Jackets' : 'CBJ',
        'Dallas Stars' : 'DAL',
        'Detroit Red Wings' : 'DET',
        'Edmonton Oilers'  : 'EDM',
        'Florida Panthers' : 'FLA',
        'Los Angeles Kings' : 'LAK',
        'Minnesota Wild' : 'MIN',
        'Montreal Canadiens' : 'MTL',
        'Nashville Predators' : 'NSH',
        'New Jersey Devils' : 'NJD',
        'New York Islanders' : 'NYI',
        'New York Rangers' : 'NYR',
        'Ottawa Senators' : 'OTT',
        'Philadelphia Flyers' : 'PHI',
        'Pittsburgh Penguins' : 'PIT',
        'San Jose Sharks' : 'SJS',
        'St. Louis Blues' : 'STL',
        'Tampa Bay Lightning' : 'TBL',
        'Toronto Maple Leafs' : 'TOR',
        'Vancouver Canucks' : 'VAN',
        'Vegas Golden Knights' : 'VGK',
        'Washington Capitals' : 'WSH',
        'Winnipeg Jets' : 'WPG'}

# Map the names
df["awayTeam"] = df["awayTeam"].map(conv)
df["homeTeam"] = df["homeTeam"].map(conv)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>awayTeam</th>
      <th>awayGoals</th>
      <th>homeTeam</th>
      <th>homeGoals</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-02</th>
      <td>VAN</td>
      <td>2.0</td>
      <td>EDM</td>
      <td>3.0</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>WSH</td>
      <td>3.0</td>
      <td>STL</td>
      <td>2.0</td>
      <td>OT</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>OTT</td>
      <td>3.0</td>
      <td>TOR</td>
      <td>5.0</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>SJS</td>
      <td>1.0</td>
      <td>VGK</td>
      <td>4.0</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>ARI</td>
      <td>1.0</td>
      <td>ANA</td>
      <td>2.0</td>
      <td>Reg</td>
    </tr>
  </tbody>
</table>
</div>



These games, sadly, have not been played, so I am going to drop them as well.


```python
# End of season hasn't happened :(
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>awayTeam</th>
      <th>awayGoals</th>
      <th>homeTeam</th>
      <th>homeGoals</th>
      <th>Extra</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-04</th>
      <td>CHI</td>
      <td>NaN</td>
      <td>NYR</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>PIT</td>
      <td>NaN</td>
      <td>OTT</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>ANA</td>
      <td>NaN</td>
      <td>SJS</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>MTL</td>
      <td>NaN</td>
      <td>TOR</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>VGK</td>
      <td>NaN</td>
      <td>VAN</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
  </tbody>
</table>
</div>




```python
# First save the season schedule into a separate df
schedule = df[["homeTeam", "awayTeam"]].copy()

# Then drop all rows with NaN, that is, those without scores
df.dropna(inplace=True)
```

Now I will uniquely label each team. I first build a dummy pandas dataframe to map team abbreviations to a unique index, then I perform a series of joins using [pandas's merge method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) to label both the home and away teams. For this array, I will also figure out how many games each team has played.


```python
allTeams = df["homeTeam"].unique()
teams = pd.DataFrame(allTeams, columns=['name'])
teams['ind'] = teams.index

# Figure out name of home team to help identify unique games
def calcGames(df, name):
    return (df['homeTeam'] == name).sum() + (df['awayTeam'].values == name).sum()

teams['gamesPlayed'] = teams.apply(lambda x : calcGames(df, x["name"]), axis=1)
```


```python
teams.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>ind</th>
      <th>gamesPlayed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EDM</td>
      <td>0</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>STL</td>
      <td>1</td>
      <td>71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TOR</td>
      <td>2</td>
      <td>70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VGK</td>
      <td>3</td>
      <td>71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ANA</td>
      <td>4</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.merge(df, teams, left_on='homeTeam', right_on='name', how='left')
df = df.rename(columns={'ind': 'homeIndex'}).drop(columns=["name", "gamesPlayed"])
df = pd.merge(df, teams, left_on = 'awayTeam', right_on = 'name', how = 'left')
df = df.rename(columns = {'ind': 'awayIndex'}).drop(columns=["name", "gamesPlayed"])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>awayTeam</th>
      <th>awayGoals</th>
      <th>homeTeam</th>
      <th>homeGoals</th>
      <th>Extra</th>
      <th>homeIndex</th>
      <th>awayIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VAN</td>
      <td>2.0</td>
      <td>EDM</td>
      <td>3.0</td>
      <td>Reg</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WSH</td>
      <td>3.0</td>
      <td>STL</td>
      <td>2.0</td>
      <td>OT</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OTT</td>
      <td>3.0</td>
      <td>TOR</td>
      <td>5.0</td>
      <td>Reg</td>
      <td>2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SJS</td>
      <td>1.0</td>
      <td>VGK</td>
      <td>4.0</td>
      <td>Reg</td>
      <td>3</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARI</td>
      <td>1.0</td>
      <td>ANA</td>
      <td>2.0</td>
      <td>Reg</td>
      <td>4</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



Our season data is looking pretty good.

For the model criticism and examination we'll perform later, I first want to compute things like points percentage (what fraction of points did a team earn in all their games), games played, goals for per game, and goals against per game. I expect these quantities to obviously correlate with defense and attack strengths. Note that for games that end in a SO, I follow the NHL convention and count it as a goal scored by the winning team.


```python
# Felt lazy so I typed the values in by-hand from https://www.nhl.com/standings
points = {'ANA' : 67,
          'ARI' : 74,
          'BOS' : 100,
          'BUF' : 68,
          'CGY' : 79,
          'CAR' : 81,
          'CHI' : 72,
          'COL' : 92,
          'CBJ' : 81,
          'DAL' : 82,
          'DET' : 39,
          'EDM' : 83,
          'FLA' : 78,
          'LAK' : 64,
          'MIN' : 77,
          'MTL' : 71,
          'NSH' : 78,
          'NJD' : 68,
          'NYI' : 80,
          'NYR' : 79,
          'OTT' : 62,
          'PHI' : 89,
          'PIT' : 86,
          'SJS' : 63,
          'STL' : 94,
          'TBL' : 92,
          'TOR' : 81,
          'VAN' : 78,
          'VGK' : 86,
          'WSH' : 90,
          'WPG' : 80}

# Compute points percentage for each team
pointsArr = np.empty(len(teams))
for ii, team in enumerate(teams["name"]):
    pointsArr[ii] = points[team]/(int(teams[teams.name == team]["gamesPlayed"].values) * 2)

# First I'll create groups for away and home teams
awayGroup = df.groupby("awayTeam")
homeGroup = df.groupby("homeTeam")

# Calculate goals for per game
scoredTeam = awayGroup.sum()["awayGoals"] + homeGroup.sum()["homeGoals"]
scored = teams.join(scoredTeam.to_frame(), on="name", how="left")
scored.columns = ["name", "ind", "gamesPlayed", "goalsFor"]
goalsFor = scored["goalsFor"].values/scored["gamesPlayed"].values

# Calculate goals against per game
concededTeam = awayGroup.sum()["homeGoals"] + homeGroup.sum()["awayGoals"]
conceded = teams.join(concededTeam.to_frame(), on="name", how="left")
conceded.columns = ["name", "ind", "gamesPlayed", "goalsAgainst"]
goalsAgainst = (conceded["goalsAgainst"]/conceded["gamesPlayed"]).values
```

Our data is now ready! Below, I will describe the mathematical model we use that was developed by [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) and extended by [Daniel Weitzenfeld](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/).

### Define the mathematical model

---

Here I describe how we will mathematically model the games. From [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf), we will model the number of observed goals in the gth game for the jth team as a Poisson model:

$y_{g,j} | \theta_{g,j} = \mathrm{Poisson}(\theta_{g,j})$ for observed goals $y$ in the gth game for the jth team.

In this equation, $\theta_{g}=(\theta_{g,h}, \theta_{g,a})$ represent "the scoring intensity" for the given team in the given game. Note, j = h indicates the home team whereas j = a indicates the away team.

[Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) and [Daniel Weitzenfeld](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/) use a log-linear model for $\theta$ that is decomposed into latent, or unobserved, terms for the home ice advantage (home), a team's attacking strength (att), a team's defensive strength (def), and an intercept term (intercept) that Daniel Weitzenfeld uses to capture the the mean number of goals scored by a team. Therefore, the home team's attacking ability, $att_{h(g)}$, is pitted against the away team's defensive ability, $def_{a(g)}$ where $h(g)$ and $a(g)$ identify which teams are the home and away team in the gth game. A strong attacking team will have a large $att$, whereas a good defensive team will have a large negative $def$.

Note that to maintain model identifiability, we follow [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) to enforce a "sum-to-zero" constraint on both att and def. Below, I will show how to do this with pymc3. This constraint, coupled with the fact that we are using a linear model, will allow us to directly compare the team abilities we will infer.

Putting this all together, our model for the home and away log scoring intensity is as follows:

$\log{\theta_{h,g}} = intercept + home + att_{h(g)} + def_{a(g)}$ for the home team and

$\log{\theta_{a,g}} = intercept + att_{a(g)} + def_{h(g)}$ for the away team.

Note how the team indicies are reversed in the two equations based on our assumption of a log-linear model for how the team stength parameters interact. The scoring intensity of the away team, $\log{\theta_{a,g}}$, for example, depends on the sum of the away team's attacking strength, $att_{a(g)}$, home team's defensive ability, $def_{h(g)}$, and the typical amount of goals scored by a team, the intercept.

### Define the (hyper)priors

---

All Bayesian models require prior and hyperprior distributions for the model parameters and hyperparameters, respectively. I adopt the priors used by both [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) and [Daniel Weitzenfeld](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/) and I list them below for completeness.

Note that Normal distributions in pymc3 are initialized with a mean, $\mu$, and a precision, $\tau$, instead of the standard mean and variance, $\sigma^2$. Therefore, a Normal distribution with a small $\tau = 0.0001$ approximates a Uniform distribution with effectively infinite bounds. Also, here I will use $t$ to index an arbitrary team.

The flat priors for the home and intercept terms are given by

$home \sim \mathrm{Normal}(0,0 .0001)$

$intercept \sim \mathrm{Normal}(0, 0.0001)$

The hyperpriors for each team's attacking and defensive strengths are

$att_t \sim \mathrm{Normal}(0, \tau_{att})$

$def_t \sim \mathrm{Normal}(0, \tau_{def})$

where we neglect the mean terms because of our "sum-to-zero" constraint. The hyperpriors on the precisions are given by

$\tau_{att} \sim \mathrm{Gamma}(0.1, 0.1)$

$\tau_{def} \sim \mathrm{Gamma}(0.1, 0.1)$

We assume that each team's attacking and defensive strengths are assumed to be drawn the same parent distribution and are hence exchangeable.

### Build the PyMC3 model

---

Now let us use PyMC3 to build the model using probabilistic programming. After I define the model, I can draw samples from the posterior distribution.

First, I will define a few useful quantities to make coding up the model easier.


```python
# Observed goals
observedHomeGoals = df["homeGoals"].values
observedAwayGoals = df["awayGoals"].values

# Inidices for home and away teams
homeTeam = df["homeIndex"].values
awayTeam = df["awayIndex"].values

# Number of teams, games
numTeams = len(list(set(df["homeIndex"])))
numGames = len(df)
```


```python
with pm.Model() as model:

    # Home, intercept priors
    home = pm.Normal('home', mu=0.0, tau=0.0001)
    intercept = pm.Normal('intercept', mu=0.0, tau=0.0001)

    # Hyperpriors on taus
    tauAtt = pm.Gamma("tauAtt", alpha=0.1, beta=0.1)
    tauDef = pm.Gamma("tauDef", alpha=0.1, beta=0.1)

    # Attacking, defensive strength for each team
    attsStar = pm.Normal("attsStar", mu=0.0, tau=tauAtt, shape=numTeams)
    defsStar = pm.Normal("defsStar", mu=0.0, tau=tauDef, shape=numTeams)

    # Impose "sum-to-zero" constraint
    atts = pm.Deterministic('atts', attsStar - tt.mean(attsStar))
    defs = pm.Deterministic('defs', defsStar - tt.mean(defsStar))

    # Compute theta for the home and away teams
    homeTheta = tt.exp(intercept + home + atts[homeTeam] + defs[awayTeam])
    awayTheta = tt.exp(intercept + atts[awayTeam] + defs[homeTeam])

    # Assume a Poisson likelihood for the observed goals
    homeGoals = pm.Poisson('homeGoals', mu=homeTheta, observed=observedHomeGoals)
    awayGoals = pm.Poisson('awayGoals', mu=awayTheta, observed=observedAwayGoals)
```

### Sample the posterior distribution and examine MCMC convergence

---

With this probabilistic model in hand, let's sample from the posterior distribution. Once the sampling is completed, I will examine numerous diagnostic statistics, including the Gelman-Rubin statistic, and visually examine the joint and marginal posterior distribution to confirm convergence.


```python
with model:
    trace = pm.sample(draws=10000, tune=1000, progressbar=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [defsStar, attsStar, tauDef, tauAtt, intercept, home]
    Sampling 2 chains, 0 divergences: 100%|██████████| 20200/20200 [02:26<00:00, 138.00draws/s]
    The acceptance probability does not match the target. It is 0.9367762282430656, but should be close to 0.8. Try to increase the number of tuning steps.
    The acceptance probability does not match the target. It is 0.9298682807628864, but should be close to 0.8. Try to increase the number of tuning steps.



```python
pm.traceplot(trace, var_names=['intercept', 'home', 'tauAtt', 'tauDef']);
```


![png](bayesianNHL_files/bayesianNHL_34_0.png)



```python
pm.summary(trace, var_names=['intercept', 'home', 'tauAtt', 'tauDef'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>intercept</th>
      <td>1.052</td>
      <td>0.018</td>
      <td>1.018</td>
      <td>1.086</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>20293.0</td>
      <td>20293.0</td>
      <td>20286.0</td>
      <td>14523.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>home</th>
      <td>0.084</td>
      <td>0.025</td>
      <td>0.037</td>
      <td>0.131</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>20294.0</td>
      <td>18719.0</td>
      <td>20305.0</td>
      <td>12261.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>tauAtt</th>
      <td>51.271</td>
      <td>15.266</td>
      <td>24.746</td>
      <td>79.821</td>
      <td>0.097</td>
      <td>0.071</td>
      <td>24849.0</td>
      <td>23229.0</td>
      <td>24757.0</td>
      <td>16822.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>tauDef</th>
      <td>66.623</td>
      <td>19.002</td>
      <td>34.314</td>
      <td>103.936</td>
      <td>0.117</td>
      <td>0.085</td>
      <td>26570.0</td>
      <td>25032.0</td>
      <td>26125.0</td>
      <td>16042.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
varNames = ['intercept', 'home', 'tauAtt', 'tauDef']

samples = np.empty((20000, 4))
for ii, var in enumerate(varNames):
    samples[:,ii] = trace[var]

_ = corner.corner(samples, labels=varNames, lw=2, hist_kwargs={"lw" : 2}, show_titles=True)
```


![png](bayesianNHL_files/bayesianNHL_36_0.png)


It appears that the only significant correlations are between the intercept term and the home ice advantage term. This correlation makes sense, however, given that the intercept effectively quantifies the typical number of log goals (typical goals = exp(intercept)) scored by a team. If the average team scores more goals, we would expect the home ice advantage to weaken.

Now let's consider the Bayesian fraction of mission information (BFMI), the Gelman-Rubin statistic, and the marginal energy distribution of the MCMC. I won't go into the mathetmatics here, but we want the BFMI and Gelman-Rubin statistics to be about 1 for a converged chain. Furthermore, if the distribution of marginal energy and the energy transition are similar, the chain is likely converged.


```python
# Estimate the maximum Bayesian fraction of missing information (BFMI) and
# Gelman-Rubin statistic
bfmi = np.max(pm.stats.bfmi(trace))
maxGR = max(np.max(gr) for gr in pm.stats.rhat(trace).values()).values
print("Rhats:", pm.stats.rhat(trace).values())
```

    Rhats: ValuesView(<xarray.Dataset>
    Dimensions:         (attsStar_dim_0: 31, atts_dim_0: 31, defsStar_dim_0: 31, defs_dim_0: 31)
    Coordinates:
      * attsStar_dim_0  (attsStar_dim_0) int64 0 1 2 3 4 5 6 ... 25 26 27 28 29 30
      * defsStar_dim_0  (defsStar_dim_0) int64 0 1 2 3 4 5 6 ... 25 26 27 28 29 30
      * atts_dim_0      (atts_dim_0) int64 0 1 2 3 4 5 6 7 ... 24 25 26 27 28 29 30
      * defs_dim_0      (defs_dim_0) int64 0 1 2 3 4 5 6 7 ... 24 25 26 27 28 29 30
    Data variables:
        home            float64 1.0
        intercept       float64 1.001
        attsStar        (attsStar_dim_0) float64 1.0 0.9999 1.0 1.0 ... 1.0 1.0 1.0
        defsStar        (defsStar_dim_0) float64 1.0 1.0 1.0 1.001 ... 1.0 1.0 1.0
        tauAtt          float64 1.0
        tauDef          float64 1.0
        atts            (atts_dim_0) float64 1.0 1.0 1.0 1.0 ... 1.0 1.001 1.0 1.0
        defs            (defs_dim_0) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0)



```python
ax = pm.energyplot(trace, kind="histogram", legend=True, figsize=(6, 4))
ax.set_title("BFMI = %lf\nGelman-Rubin = %lf" % (bfmi, maxGR));
```


![png](bayesianNHL_files/bayesianNHL_40_0.png)


It looks like our model has converged!

### Explore model implications

---

Now that I convinced myself that we drew enough valid samples from the posterior distribution, I can examine the inferred posterior distributions for our model latent variables like a team's offensive and defensive strengths.


```python
ax = pm.forestplot(trace, var_names=['atts'])
ax[0].set_yticklabels(teams.iloc[::-1]['name'].tolist())
ax[0].axvline(0, color="k", zorder=0, ls="--")
ax[0].set_xlabel('Team Offensive Strength', fontsize=15)
ax[0].set_title("");
```


![png](bayesianNHL_files/bayesianNHL_43_0.png)



```python
ax = pm.forestplot(trace, var_names=['defs'])
ax[0].axvline(0, color="k", zorder=0, ls="--")
ax[0].set_yticklabels(teams.iloc[::-1]['name'].tolist())
ax[0].set_xlabel('Team Defensive Strength', fontsize=15)
ax[0].set_title("");
```


![png](bayesianNHL_files/bayesianNHL_44_0.png)


Now let's plot these same quantities, but ranking teams from worst (Detroit) to best. Remember: a team wants to have a large attack strength (score more goals!) and a large *negative* defense strength (make the other team score fewer goals!).


```python
# Calculate median, 68% CI for atts, defs for each team
medAtts = np.median(trace["atts"], axis=0)
medDefs = np.median(trace["defs"], axis=0)

defsCI = pm.stats.hpd(trace["defs"], credible_interval=0.68)
attsCI = pm.stats.hpd(trace["atts"], credible_interval=0.68)
```


```python
# Plot ordered attacking strength
fig, ax = plt.subplots(figsize=(10,4))

# Order values by worst to best attacking
inds = np.argsort(medAtts)

x = np.arange(len(medAtts))
ax.errorbar(x, medAtts[inds], yerr=[medAtts[inds] - attsCI[inds,0], attsCI[inds,1] - medAtts[inds]],
            fmt='o')

ax.axhline(0, lw=2, ls="--", color="k", zorder=0)
ax.set_title('68% Confidence Interval of Attack Strength, by Team')
ax.set_xlabel('Team')
ax.set_ylabel('Posterior Attack Strength\n(Above 0 = Good)')
_ = ax.set_xticks(x)
_ = ax.set_xticklabels(teams["name"].values[inds], rotation=45)
```


![png](bayesianNHL_files/bayesianNHL_47_0.png)



```python
# Plot ordered defense strength
fig, ax = plt.subplots(figsize=(10,4))

# Order values by worst to best attacking
inds = np.argsort(medDefs)[::-1]

x = np.arange(len(medDefs))
ax.errorbar(x, medDefs[inds], yerr=[medDefs[inds] - defsCI[inds,0], defsCI[inds,1] - medDefs[inds]],
            fmt='o')

ax.axhline(0, lw=2, ls="--", color="k", zorder=0)
ax.set_title('68% Confidence Interval of Defense Strength, by Team')
ax.set_xlabel('Team')
ax.set_ylabel('Posterior Defense Strength\n(Below 0 = Good)')
_ = ax.set_xticks(x)
_ = ax.set_xticklabels(teams["name"].values[inds], rotation=45)
```


![png](bayesianNHL_files/bayesianNHL_48_0.png)


Below, I'll see how teams' points percentage varies as a function of attack and defense strengths. I expect that teams with a higher points percentage, e.g. BOS and STL, will have large attack and large negative defense strengths and for the converse to be true for bad teams like DET.


```python
fig, ax = plt.subplots(figsize=(6,5))

im = ax.scatter(medDefs, medAtts, c=pointsArr, s=60, zorder=1)
ax.axhline(0, lw=1.5, ls="--", color="grey", zorder=0)
ax.axvline(0, lw=1.5, ls="--", color="grey", zorder=0)

cbar = fig.colorbar(im)
cbar.set_label(r"Points % as of March 12$^{\mathrm{th}}$", fontsize=15)
ax.set_xlabel('Posterior Defense Strength', fontsize=15)
ax.set_ylabel('Posterior Attack Strength', fontsize=15)
ax.set_xlim(-0.2, 0.2);
ax.set_ylim(-0.32, 0.15);
```


![png](bayesianNHL_files/bayesianNHL_50_0.png)


Above I plotted each team's posterior attack strength vs. the posterior defense strength. Each point represents a team and the color encodes the team's season points total as of March 12th, 2020 when the NHL season was officially suspended. Clearly, there is a gradient in total points (color) that follows a reasonable trend: teams with strong attacks (large attack strength) and strong defense (large negative defense strength) tend to perform bettern and accumulate more points. BOS, arguably the best team in the NHL in 2019-2020, is located in the strong attack/defense quandrant and is appropriately colored yellow for 100 points. DET (purple dot in the bad quadrant), on the other hand, is far-and-away the worst team in the NHL and our model captures that.


```python
fig, ax = plt.subplots(figsize=(6,5))

im = ax.scatter(medAtts, goalsFor, c=pointsArr, s=60, zorder=1)
ax.axhline(np.mean(goalsFor), lw=1.5, ls="--", color="grey", zorder=0)
ax.axvline(0, lw=1.5, ls="--", color="grey", zorder=0)

cbar = fig.colorbar(im)
cbar.set_label(r"Point % as of March 12$^{\mathrm{th}}$", fontsize=15)
ax.set_xlabel('Posterior Attack Strength', fontsize=15)
ax.set_ylabel('Goals For Per Game', fontsize=15);
```


![png](bayesianNHL_files/bayesianNHL_52_0.png)


It looks like our posterior attack strength parameter accurately captures offense strength as there is a tight correlation between the two. Generally, it appears the number of points a team earns increases with both posterior attack strength and goals for as we'd expect, but there is some scatter that is likely caused by the uncertainty in the posterior distributions.


```python
fig, ax = plt.subplots(figsize=(6,5))

im = ax.scatter(medAtts, goalsAgainst, c=pointsArr, s=60, zorder=1)
ax.axhline(np.mean(goalsAgainst), lw=1.5, ls="--", color="grey", zorder=0)
ax.axvline(0, lw=1.5, ls="--", color="grey", zorder=0)

cbar = fig.colorbar(im)
cbar.set_label(r"Point % as of March 12$^{\mathrm{th}}$", fontsize=15)
ax.set_xlabel('Posterior Defense Strength', fontsize=15)
ax.set_ylabel('Goals Against Per Game', fontsize=15);
```


![png](bayesianNHL_files/bayesianNHL_54_0.png)


Interestingly, the correlation between goals against and posterior defense strength is *much* weaker than what we saw above for attacking values. The correlation between points and a linear combination of goals against and posterior defense strength, i.e. the color gradient from the top left to the bottom right of the figure, appears much stronger. Perhaps there's more to defense than simply goals conceded. I think this makes sense in a game like hockey because a team can get shelled 50-15 in terms of shots but only lose 1-0 whereas a team can lose 3-0 but take more shots than the other team and dominate the play. Hockey is a non-linear game to say the least, but I think our model is doing pretty well given its simplicity.

### Simulating games (and the rest of the 2019-2020 season)

---

I have shown with fairly high confidence that the defending champions, the St. Louis Blues, are a much better offensive and defensive team than the Chicago Blackhawks. Also, Detroit is a bad team overall (other than Robby Fabbri and Dylan Larkin). Therefore, it appears that my model is realistically modeling the strengths and weaknesses of NHL teams given the games we have observed so far and our simplified model. Now we can turn to making some predictions using simulations.

My favorite aspect of probabilistic modeling is how I can draw samples from the posterior distribution to simulate games and reasonably account for model uncertainty. That is, I can estimate the likelihood that NHL Team A beats Team B at home. Moreover, I can estimate probability distributions for the score. Naturally, this can be extrapolated to simulating "Best-of-7" series and the playoffs more generally. For now, I will focus on simulating individual games and then the rest of the season to see how many points each team likely would have earned throughout a full 82 game season. Recall that above, I saved the entire season's schedule. We will use that below to simulate the rest of the season.


```python
def simulateGame(trace, ind, homeTeam, awayTeam, teams, chain=0):
    """
    Simulate an NHL game where awayTeam plays at homeTeam and trace
    is a draw from the posterior distribution for model parameters,
    e.g. atts and defs. In this simplified model, if the game goes to
    OT, I say the game goes to a SO 34.4% of the team, the empirical
    fraction from this season's NHL results. If the game ends in either
    and OT or SO, I assign each team equal odds to win and randomly decide,
    assigning an extra goal to the winner.

    Parameters
    ----------
    trace : iterable
        Posterior distribution MCMC chain
    ind : int
        Index representing posterior draw
    homeTeam : str
        Name of the home team, like STL
    awayTeam : str
        Name of the away team, like CHI
    teams : pd.DataFrame
        pandas dataframe of teams that maps
        team name to a unique index
    chain : int (optional)
        Which chain to draw from. Defaults to 0.

    Returns
    -------
    homeGoals : int
        number of goals scored by the home team
    awayGoals : int
        number of goals scored by the away team
    homeWin : bool
        whether or not the hometeam won
    homePoints : int
        number of standings points earned by home team
    awayPoints : int
        number of standings points earned by away team
    note : str
        indicates if the game finished in regulation (REG),
        overtime (OT), or a shooutout (SO).
    """

    # Extract posterior parameters
    home = trace.point(ind, chain=chain)["home"]
    intercept = trace.point(ind)["intercept"]
    homeAtt = trace.point(ind)["atts"][int(teams[teams["name"] == homeTeam]["ind"])]
    homeDef = trace.point(ind)["defs"][int(teams[teams["name"] == homeTeam]["ind"])]
    awayAtt = trace.point(ind)["atts"][int(teams[teams["name"] == awayTeam]["ind"])]
    awayDef = trace.point(ind)["defs"][int(teams[teams["name"] == awayTeam]["ind"])]

    # Compute home and away goals using log-linear model, draws for model parameters
    # from posterior distribution. Recall - model goals as a draws from
    # conditionally-independent Poisson distribution: y | theta ~ Poisson(theta)
    homeTheta = np.exp(home + intercept + homeAtt + awayDef)
    awayTheta = np.exp(intercept + awayAtt + homeDef)
    homeGoals = np.random.poisson(homeTheta)
    awayGoals = np.random.poisson(awayTheta)

    # Figure out who wins
    note = "REG"
    if homeGoals > awayGoals:
        homeWin = True
        homePoints = 2
        awayPoints = 0
    elif awayGoals > homeGoals:
        homeWin = False
        awayPoints = 2
        homePoints = 0
    # Overtime!
    else:
        # Each team gets at least 1 point now
        homePoints = 1
        awayPoints = 1

        # Does the game go into a shootout?
        if np.random.uniform(low=0, high=1) < 0.344:
            note = "SO"
            # Randomly decided who wins
            if np.random.uniform(low=0, high=1) < 0.5:
                homeWin = True
                homeGoals += 1
                homePoints = 2
            else:
                homeWin = False
                awayGoals += 1
                awayPoints = 2
        # No shootout, randomly decide who wins OT
        else:
            note = "OT"
            # Randomly decided who wins
            if np.random.uniform(low=0, high=1) < 0.5:
                homeWin = True
                homeGoals += 1
                homePoints = 2
            else:
                homeWin = False
                awayGoals += 1
                awayPoints = 2

    return homeGoals, awayGoals, homeWin, homePoints, awayPoints, note  
```

### Case Study: How likely is it that the St. Louis Blues sweep the season series against the Chicago Blackhawks, a team that has not won a game in the playoffs since 2016?

---

This season, the Blues swept the season series with Chicago, 4 wins to 0 losses, for the first time in franchise history. Using my model, I can estimate how often that would have occured. For this estimation, I'll simulate 2,500 4 games series where each team has 2 home games.


```python
nTrials = 2500
nGames = 4

# numpy array to hold results
bluesRes = np.zeros((nTrials, nGames), dtype=int)

# random array of indicies to sample from
choices = np.arange(10000)

for ii in range(nTrials):

    # nGames game series
    for jj in range(nGames):

        # Set home, away team
        if jj < 2:
            homeTeam = "STL"
            awayTeam = "CHI"
        else:
            homeTeam = "CHI"
            awayTeam = "STL"

        # Draw random sample with replacement from one of 2 MCMC chains
        ind = np.random.choice(choices)
        chain = np.random.randint(2)

        # Simulate the game, save whether or not Blues won the game
        _, _, homeWin, _, _, _ = simulateGame(trace, ind, homeTeam, awayTeam, teams, chain=chain)
        if jj < 2:
            bluesRes[ii, jj] = int(homeWin)
        else:
            bluesRes[ii, jj] = int(not homeWin)
```


```python
mask = bluesRes.all(axis=1)
print("STL sweeps the season series against CHI in %0.1lf %% of simulated seasons." % (np.mean(mask) * 100))

mask = (bluesRes.sum(axis=1) > 2)
print("STL wins the season series against CHI in %0.1lf %% of simulated seasons." % (np.mean(mask) * 100))

mask = (bluesRes.sum(axis=1) == 2)
print("STL ties the season series against CHI in %0.1lf %% of simulated seasons." % (np.mean(mask) * 100))

mask = (bluesRes.sum(axis=1) < 2)
print("CHI wins the season series against STL in %0.1lf %% of simulated seasons." % (np.mean(mask) * 100))
```

    STL sweeps the season series against CHI in 9.9 % of simulated seasons.
    STL wins the season series against CHI in 42.2 % of simulated seasons.
    STL ties the season series against CHI in 35.5 % of simulated seasons.
    CHI wins the season series against STL in 22.2 % of simulated seasons.


Awesome! It was of course unlikely for the Blues to sweep the regular season series against the Blackhawks since they were not *that* bad this year, but the Blues had a decent chance, about 1 in 10, and took it. Other than the whole *global pandemic*, this was a pretty good result. Also, my model predicts that the Blues should win, or at least tie, the season series over 3/4s of the time.

### Simulating the 2019-2020 season up until the season suspension

---

Now that I think the model is working well, I'm going to perform one final validation before I simulate the final season standings and the playoffs.  I will replay the season up until it was suspended using draws from the posterior distribution. If my model is working, I'd hope that a team's expected points is nearly the same as how many points they actually scored.

I'll select which games were played and use this schedule to re-simulate the played regular season.


```python
playedSchedule = schedule.iloc[schedule.index < datetime(2020, 3, 12)]
```


```python
playedSchedule.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>homeTeam</th>
      <th>awayTeam</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-11</th>
      <td>ANA</td>
      <td>STL</td>
    </tr>
    <tr>
      <th>2020-03-11</th>
      <td>CHI</td>
      <td>SJS</td>
    </tr>
    <tr>
      <th>2020-03-11</th>
      <td>COL</td>
      <td>NYR</td>
    </tr>
    <tr>
      <th>2020-03-11</th>
      <td>EDM</td>
      <td>WPG</td>
    </tr>
    <tr>
      <th>2020-03-11</th>
      <td>LAK</td>
      <td>OTT</td>
    </tr>
  </tbody>
</table>
</div>



Now I can run some simulations and here I outline my simulation procedure. For each iteration, I will make one draw from the posterior distribution for each team, i.e. each game will be played with the same home, intercept, attack strengths, and defense strenghts that were drawn from the posterior distribution for each team (except only one draw for home and intercept terms since they are the same for each team). I will then call the simulateGame function I wrote above for each game and record an estimate for the number of points earned by each team. Each team will have a column and each row will represent a season. I'll run 1,000 simulations so I can derive reasonable marginal posterior distributions for each team's expected points.


```python
# Number of seasons to simulate
numSeasons = 1000

res = list()
for ii in range(numSeasons):

    # Draw random sample with replacement from one of 2 MCMC chains
    ind = np.random.choice(choices)
    chain = np.random.randint(2)

    # Teams start season with 0 points
    tmpPoints = np.zeros(len(teams))
    for jj in range(len(playedSchedule)):
        # Select home, away teams
        homeTeam = playedSchedule.iloc[jj]["homeTeam"]
        awayTeam = playedSchedule.iloc[jj]["awayTeam"]

        # Simulate jjth game, store results
        _, _, _, homePoints, awayPoints, _ = simulateGame(trace, ind, homeTeam, awayTeam, teams, chain=chain)

        tmpPoints[int(teams[teams["name"] == homeTeam]["ind"])] += homePoints
        tmpPoints[int(teams[teams["name"] == awayTeam]["ind"])] += awayPoints

    # Save season result
    res.append(list(tmpPoints))

# Turn simulations into a dataframe as described above
playedPoints = pd.DataFrame.from_records(res, columns=teams["name"].values)
```

# Instead, try doing this by sampling from the posterior predictive trace!

Now that we've simulated the season, I will plot the inferred posterior distribution for each team's earned standings points. First, I'll plot these quantities for the Blues, then I'll do the same for every other team.


```python
fig, ax = plt.subplots(figsize=(6,5))

ax.hist(playedPoints["STL"], color="C0", bins="auto", histtype="step", lw=2, density=True)
ax.hist(playedPoints["STL"], color="C0", bins="auto", alpha=0.6, density=True)

ax.axvline(np.median(playedPoints["STL"]), color="C1", lw=3, ls="--")
ax.axvline(points["STL"], color="k", lw=3, ls="--")

ax.text(0.025, 0.95,'Posterior points: %0.1lf' % np.median(playedPoints["STL"]),
        horizontalalignment='left', color="k",
        verticalalignment='center',
        transform = ax.transAxes)

ax.text(0.025, 0.9,'Actual points: %d' % points["STL"],
        horizontalalignment='left', color="C1",
        verticalalignment='center',
        transform = ax.transAxes)

ax.text(0.025, 0.85,r'$\Delta=$%0.1lf' % (np.median(playedPoints["STL"]) - points["STL"]),
        horizontalalignment='left',
        verticalalignment='center',
        transform = ax.transAxes)

ax.set_title("STL")
ax.set_ylabel("Posterior Density", fontsize=15)
ax.set_xlabel("Points as of March 12$^{th}$", fontsize=15);
```


![png](bayesianNHL_files/bayesianNHL_69_0.png)



```python
fig, axes = plt.subplots(ncols=6, nrows=5, sharex=True, figsize=(12, 11))

teamNames = list(teams["name"])
teamNames.remove("STL")
teamNames = list(np.sort(teamNames))

for ii, ax in enumerate(axes.flatten()):

    # Get team name
    teamName = teamNames[ii]

    # Turn off all y ticks, set common x range
    ax.set_yticklabels([])
    ax.set_xlim(30, 120)

    # Histogram of posterior points
    ax.hist(playedPoints[teamName], color="C0", bins="auto", histtype="step", lw=1.5, density=True)
    ax.hist(playedPoints[teamName], color="C0", bins="auto", alpha=0.6, density=True)

    # Plot observed, median posterior points
    ax.axvline(np.median(playedPoints[teamName]), color="k", lw=2, ls="--")
    ax.axvline(points[teamName], color="C1", lw=2, ls="--")

    # Annotate with actual points, posterior points, and the difference (delta)
    if points[teamName] < 70:
        offset = 0.6
    else:
        offset = 0

    ax.text(0.025 + offset, 0.925, '%d' % np.median(playedPoints[teamName]),
            horizontalalignment='left', color="k",
            verticalalignment='center', fontsize=11.5,
            transform = ax.transAxes)
    ax.text(0.025 + offset, 0.8, '%d' % points[teamName],
            horizontalalignment='left', color="C1",
            verticalalignment='center', fontsize=11.5,
            transform = ax.transAxes)
    ax.text(0.025 + offset, 0.7 ,r'$\Delta=$%d' % (np.median(playedPoints[teamName]) - points[teamName]),
            horizontalalignment='left', fontsize=11.5,
            verticalalignment='center',
            transform = ax.transAxes)
    ax.set_title(teamName)

# Format
axes[2,0].set_ylabel("Posterior Density", fontsize=25, position=(0, 0.5));
axes[4,2].set_xlabel("Points as of March 12$^{th}$", fontsize=25, position=(1, 0));
```


![png](bayesianNHL_files/bayesianNHL_70_0.png)


It appears that my model does a pretty good job at reproducing the 2019-2020 regular season! The median posterior points predicted by my model is dead on for WPG, VAN, NYR, ANA, and CAR. Moreover, it is within a few points many teams, so I think that I can conclude that my model is actually picking up on how good teams are and appropriately modeling the results of games for this season.

It does appear, however, that my model underpredicts the points earned by the best teams, e.g. BOS, WSH, and STL. Furthermore, my model overpredicts the points earned by the worst team, DET. This effect is a consequence of the well-known effect of **shrinkage**. Shrinkage oftens occurs for hierarchical Bayesian models and both [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) and [Daniel Weitzenfeld's great write-up](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/) describe this effect in detail, so please see their write-ups for an in-depth discussion.

Basically, what's happening is that our hyperprior for attack and defense strengths has a prior mean of 0. This in effect pulls each teams strengths towards 0, depending on their observed scoring rates, effectively acting as a regularization term. [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) explain how using different hyperpriors for different classes of teams, e.g. one for the good teams, one for the average teams, and one for DET, can mitigate shrinkage. Personally, I like including forms of regularization in my models to prevent observing too many extreme results, so I will keep it in.

### Simulating the unplayed games of the 2019-2020 NHL season

---

I am convinced that my model does a good job of reproducing the results of the 2019-2020 NHL season up until the season was suspended. Moreover, the latent parameters inferred by my model, i.e. each team's attack and defense strength, appears to provide reasonable approximations for a team's scoring and defensive performance. Given a working model, I can now use it to predict the final standings for the season to see where teams would end up in the playoffs! To do that, I will simulate the rest of the season, similar to my simulations above, and add those results to the observed points each team had accrued so far. Then, for simplicity, I will sort teams by point percentage to determine who makes the playoffs and what seed they earned. Note that future work should more robustly track game-by-game results to figure out tie breaker scenarios. Note that my model can do that, I'm just taking a simpler approach.


```python
futureSchedule = schedule.iloc[schedule.index >= datetime(2020, 3, 12)]
```


```python
futureSchedule.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>homeTeam</th>
      <th>awayTeam</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-04</th>
      <td>NYR</td>
      <td>CHI</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>OTT</td>
      <td>PIT</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>SJS</td>
      <td>ANA</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>TOR</td>
      <td>MTL</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>VAN</td>
      <td>VGK</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number of seasons to simulate
numSeasons = 100

res = list()
for ii in range(numSeasons):

    # Draw random sample with replacement from one of 2 MCMC chains
    ind = np.random.choice(choices)
    chain = np.random.randint(2)

    # Teams start season with 0 points
    tmpPoints = np.zeros(len(teams))
    for jj in range(len(futureSchedule)):
        # Select home, away teams
        homeTeam = futureSchedule.iloc[jj]["homeTeam"]
        awayTeam = futureSchedule.iloc[jj]["awayTeam"]

        # Simulate jjth game, store results
        _, _, _, homePoints, awayPoints, _ = simulateGame(trace, ind, homeTeam, awayTeam, teams, chain=chain)

        tmpPoints[int(teams[teams["name"] == homeTeam]["ind"])] += homePoints
        tmpPoints[int(teams[teams["name"] == awayTeam]["ind"])] += awayPoints

    # Save season result
    res.append(list(tmpPoints))

# Turn simulations into a dataframe as described above
futurePoints = pd.DataFrame.from_records(res, columns=teams["name"].values)
```

We've simulated the rest of the season. Now, we can add each team's earned points to the posterior future points to estimate a posterior probability distribution for how many points a team would have earned by the end of the season. Then, I can compute the mean of these distribution and sort the teams to see who would have likely earned the President's Trophy.


```python
# Calculate posterior for end of season points
totalPoints = futurePoints.copy()
teamNames = list(teams["name"])
teamNames = list(np.sort(teamNames))

for team in teamNames:
    totalPoints[team] += points[team]
```


```python
meanTotal = totalPoints.mean(axis=0).sort_values(ascending=False)
```


```python
meanTotal
```




    BOS    115.49
    TBL    108.15
    COL    107.80
    WSH    106.81
    STL    106.65
    PHI    104.74
    PIT    100.89
    VGK     97.99
    DAL     97.39
    CAR     97.34
    EDM     95.64
    TOR     94.98
    NYI     94.91
    CBJ     92.85
    VAN     92.47
    FLA     92.38
    CGY     92.28
    WPG     91.49
    NSH     91.27
    NYR     91.23
    MIN     90.86
    ARI     87.87
    CHI     85.15
    MTL     82.75
    BUF     80.76
    NJD     80.29
    ANA     77.42
    LAK     75.75
    SJS     72.98
    OTT     70.90
    DET     44.85
    dtype: float64



It looks like BOS would have won the Presidents' Trophy. Congrats! Pretty much like winning the Cup, right? I am a bit sad, but ultimately not surprised, to see COL finish ahead of STL. COL was a wagon over the stretch and it would have been a tight finish with STL.

### Who Makes the Playoffs?

---




```python
# Divisions
central = ["STL", "CHI", "MIN", "DAL", "NSH", "WPG", "COL"]
pacific = ["SJS", "LAK", "ANA", "CGY", "EDM", "VGK", "VAN", "ARI"]
metro = ["WSH", "PHI", "PIT", "CAR", "CBJ", "NYI", "NYR", "NJD"]
atlantic = ["BOS", "TBL", "TOR", "FLA", "MTL", "BUF", "OTT", "DET"]

# Conferences
west = central + pacific
east = metro + atlantic

# Compute posterior standings
centralStandings = totalPoints[central].mean(axis=0).sort_values(ascending=False)
pacificStandings = totalPoints[pacific].mean(axis=0).sort_values(ascending=False)
metroStandings = totalPoints[metro].mean(axis=0).sort_values(ascending=False)
atlanticStandings = totalPoints[atlantic].mean(axis=0).sort_values(ascending=False)
westStandings = totalPoints[west].mean(axis=0).sort_values(ascending=False)
eastStandings = totalPoints[east].mean(axis=0).sort_values(ascending=False)
```

**Central Division Final Standings**


```python
centralStandings
```




    COL    107.80
    STL    106.65
    DAL     97.39
    WPG     91.49
    NSH     91.27
    MIN     90.86
    CHI     85.15
    dtype: float64



**Pacific Division Final Standings**


```python
pacificStandings
```




    VGK    97.99
    EDM    95.64
    VAN    92.47
    CGY    92.28
    ARI    87.87
    ANA    77.42
    LAK    75.75
    SJS    72.98
    dtype: float64



**Metro Division Final Standings**


```python
metroStandings
```




    WSH    106.81
    PHI    104.74
    PIT    100.89
    CAR     97.34
    NYI     94.91
    CBJ     92.85
    NYR     91.23
    NJD     80.29
    dtype: float64



**Atlantic Division Final Standings**


```python
atlanticStandings
```




    BOS    115.49
    TBL    108.15
    TOR     94.98
    FLA     92.38
    MTL     82.75
    BUF     80.76
    OTT     70.90
    DET     44.85
    dtype: float64



**Western Conference Final Standings**


```python
westStandings
```




    COL    107.80
    STL    106.65
    VGK     97.99
    DAL     97.39
    EDM     95.64
    VAN     92.47
    CGY     92.28
    WPG     91.49
    NSH     91.27
    MIN     90.86
    ARI     87.87
    CHI     85.15
    ANA     77.42
    LAK     75.75
    SJS     72.98
    dtype: float64



The Western Conference 1 and 2 wildcard teams are CGY and WPG, respectively.

**Eastern Conference Final Standings**


```python
eastStandings
```




    BOS    115.49
    TBL    108.15
    WSH    106.81
    PHI    104.74
    PIT    100.89
    CAR     97.34
    TOR     94.98
    NYI     94.91
    CBJ     92.85
    FLA     92.38
    NYR     91.23
    MTL     82.75
    BUF     80.76
    NJD     80.29
    OTT     70.90
    DET     44.85
    dtype: float64



The Eastern Conference 1 and 2 wildcard teams are CAR and NYI, respectively.


```python

```
