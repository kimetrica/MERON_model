# ----------------------------------------------------
# Calculate the coefficient of determination including
# including intercept and not including intercept
# No intercept corresponds to one-to-one line.
# Intercept corresponds to OLS fit
# ----------------------------------------------------
r_pearson, p_val = pearsonr(y_pred, y)
r_squared_intercept = r_pearson**2
r_squared_no_intercept = skl.metrics.r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Linear regression
reg_params = scipy.stats.linregress(y_pred, y)

self.logger.info('\n')
self.logger.info(f'~~~~~~~~~~~~~~{name}~~~~~~~~~~~~~~~~~~~~~~')
self.logger.info('\n')
self.logger.info('**********************************************************')
self.logger.info(
    f'Coefficient of Determination (include intercept) = {r_squared_intercept:0.3}'
)
self.logger.info(
    f'Coefficient of Determination (no intercept)      = {r_squared_no_intercept:0.3}'
)
self.logger.info(f'RMSE                             = {rmse:0.3}')
self.logger.info('**********************************************************')

y_max = np.max([np.max(y), np.max(y_pred)])
y_min = np.min([np.min(y), np.min(y_pred)])

fig, ax = plt.subplots()
sc = ax.scatter(
    y_pred, y, s=10, edgecolors="k", color=config.ki_color
)
ax.set_ylabel(r"Survey Value Total CE", fontsize=16)
ax.set_xlabel(r"y_pred Value of Total CE", fontsize=16)
ax.set_title('South Sudan', fontsize=32)
ax.text(
    0.1, 0.85, r'$r^{2}$ (intercept) = '+f'{r_squared_intercept:0.3}', {'color': 'r'},
    ha='left', va='center', transform=ax.transAxes
)
ax.text(
    0.1, 0.9, r'$r^{2}$ (no intercept) = '+f'{r_squared_no_intercept:0.3}',
    {'color': 'k'}, ha='left', va='center', transform=ax.transAxes
)
ax.text(
    0.1, 0.8, f'RMSE = {rmse:0.3}', ha='left', va='center', transform=ax.transAxes
)
ax.grid(True)
xplot_vals = np.arange(y_min, y_max + np.floor(y_max / 1))
diag_line, = ax.plot(xplot_vals,
                     xplot_vals,
                     color='k',
                     ls="--",
                     c=".15")

diag_line, = ax.plot(xplot_vals,
                     reg_params[1] + reg_params[0]*xplot_vals,
                     color='r',
                     ls="-",
                     c=".15")

ax.set_xlim([0, y_max + np.floor(y_max / 1)])
ax.set_ylim([0, y_max + np.floor(y_max / 1)])

plt.savefig(
    os.path.join(config.fig_dir, f'{name}.svg'),
    dpi=400
)
plt.clf()
