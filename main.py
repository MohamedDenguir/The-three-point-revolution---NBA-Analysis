import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
import numpy as np
from nba_api.stats.endpoints import leaguedashteamstats
import time

# -------------------------------------------------------

# Graph style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

# -------------------------------------------------------
# List of NBA seasons we want to analyze
seasons = ['2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
           '2020-21', '2021-22', '2022-23', '2023-24']
seasons_data = []

for season in seasons:
    # Request the team stats for this specific season
    data = leaguedashteamstats.LeagueDashTeamStats(season=season, season_type_all_star='Regular Season')
    # Extract the data as a table using the 'dataframe' command
    season_dataframe = data.get_data_frames()[0]
    # Add a column to remember which season this data belongs to
    season_dataframe['SEASON'] = season
    # Add this season's data to our list
    seasons_data.append(season_dataframe)
    # Wait 1 second between each request to avoid being blocked by the NBA API
    time.sleep(1)

# Combine all seasons into one single table
all_dataframe = pd.concat(seasons_data, ignore_index=True)

# -------------------------------------------------------

# Select only the columns we need for our analysis
important_columns = ['TEAM_NAME', 'SEASON', 'W', 'L', 'W_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FGM', 'FGA', 'PTS']
# Keep only those columns in our table
team_stats = all_dataframe[important_columns]
# Make a clean copy to avoid warnings from pandas
team_stats = team_stats.copy()

# Add a new column 'the share of total shots that are 3-point attempts'
team_stats['THREE_POINT_SHARE'] = team_stats['FG3A'] / team_stats['FGA']


# -------------------------------------------------------

# Plot 1 : Evolution of 3-point shooting over the seasons
# Compute the average of each stat across all 30 teams, for each season
season_averages = team_stats.groupby('SEASON')[['FG3A', 'FG3_PCT', 'THREE_POINT_SHARE']].mean().reset_index()

# Create a figure with 3 side by side graphs
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Graph 1 : how many 3-point shots were attempted on average each season
axes[0].plot(season_averages['SEASON'], season_averages['FG3A'], marker='o', color='red', linewidth=3)
axes[0].set_title('3-Point Attempts\nper Team (Average)')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('3-Point Attempts')
axes[0].tick_params(axis='x', rotation=45)

# Graph 2 : what percentage of all shots were 3-point attempts each season
axes[1].plot(season_averages['SEASON'], season_averages['THREE_POINT_SHARE'], marker='o', color='blue', linewidth=3)
axes[1].set_title('Share of 3-Pointers\nin Total Shots')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('3PT Shot Share')
axes[1].tick_params(axis='x', rotation=45)

# Graph 3 : what was the average 3-point shooting percentage each season
axes[2].plot(season_averages['SEASON'], season_averages['FG3_PCT'], marker='o', color='green', linewidth=3)
axes[2].set_title('3-Point Shooting\nPercentage')
axes[2].set_xlabel('Season')
axes[2].set_ylabel('3PT Shooting %')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle('The 3-Point Revolution in the NBA (2015-2024)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------------------

# Plot 2 : Do teams that attempt more 3-pointers win more ?
plt.figure(figsize=(10, 6))

plt.scatter(team_stats['FG3A'], team_stats['W'], alpha=0.5, color='coral', s=60)

# Compute the regression line
slope, origin, r, p, error = sp_stats.linregress(team_stats['FG3A'], team_stats['W'])

# Generate 100 evenly spaced x values to draw a smooth regression line
x = np.linspace(team_stats['FG3A'].min(), team_stats['FG3A'].max(), 100)

# Draw the regression line on the graph
plt.plot(x, slope * x + origin, color='black', linewidth=2)

# Add title and axis labels
plt.title(f'3-Point Attempts vs Wins (2015-2024)\nr = {r:.3f} | p-value = {p:.3f}', fontweight='bold')
plt.xlabel('3-Point Attempts per Season')
plt.ylabel('Wins')

plt.tight_layout()
plt.show()

# -------------------------------------------------------

# Plot 3 : Do teams that shoot better from 3 win more ?
plt.figure(figsize=(10, 6))

# Draw one dot per team per season
plt.scatter(team_stats['FG3_PCT'], team_stats['W'], alpha=0.5, color='steelblue', s=60)

# Compute the regression line for shooting percentage vs wins
shooting_slope, shooting_origin, shooting_r, shooting_p, shooting_error = sp_stats.linregress(team_stats['FG3_PCT'], team_stats['W'])

# Generate 100 evenly spaced x values to draw a smooth regression line
shooting_pct_x_line = np.linspace(team_stats['FG3_PCT'].min(), team_stats['FG3_PCT'].max(), 100)

# Draw the regression line on the graph
plt.plot(shooting_pct_x_line, shooting_slope * shooting_pct_x_line + shooting_origin, color='black', linewidth=2)

# Add title and axis labels
plt.title(f'3-Point Shooting % vs Wins (2015-2024)\nr = {shooting_r:.3f} | p-value = {shooting_p:.3f}', fontweight='bold')
plt.xlabel('3-Point Shooting %')
plt.ylabel('Wins')

plt.tight_layout()
plt.show()

# -------------------------------------------------------

# Plot 4 : Has the 3-point shot become more decisive over time ?
# Empty list to store the correlation result for each season
yearly_correlations = []

# Loop through each season and compute the correlation separately
for season in seasons:
    # Keep only the rows for this specific season
    season_data = team_stats[team_stats['SEASON'] == season]
    # Compute the correlation between 3PT shooting % and wins for this season
    r_value, p_value = sp_stats.pearsonr(season_data['FG3_PCT'], season_data['W'])
    # Store the season and its correlation value
    yearly_correlations.append({'SEASON': season, 'CORRELATION': r_value})

# Convert the list into a table
correlation_by_year = pd.DataFrame(yearly_correlations)

# Draw a bar chart showing the correlation for each season
plt.figure(figsize=(10, 6))
plt.bar(correlation_by_year['SEASON'], correlation_by_year['CORRELATION'], color='royalblue', alpha=0.8)

plt.title('Correlation Between 3PT Shooting % and Wins per Season', fontweight='bold')
plt.xlabel('Season')
plt.ylabel('Correlation (r)')
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------

# Summary
print("---Summary---")
print(f"3PT Attempts vs Wins: r = {r:.3f} | p-value = {p:.3f}")
print(f"3PT Shooting % vs Wins: r = {shooting_r:.3f} | p-value = {shooting_p:.3f}")
print("Conclusion: 3PT efficiency appears substantially more correlated with wins than shooting volume.")
