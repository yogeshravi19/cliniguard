import json

# 1. Load the original Colab notebook
with open('CLINIGUARD_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Keep cells from 0 to 9 (Intro, dependencies, loading datasets, visualizing datasets, formulas, extracting features, visualizing signals)
new_cells = nb['cells'][:10]

# Add imports for MLP in the imports cell (cell 2)
imports_cell = new_cells[2]
import_src = imports_cell['source']
# Ensure MLPClassifier is imported if not already
found_mlp = any("MLPClassifier" in line for line in import_src)
if not found_mlp:
    for i, line in enumerate(import_src):
        if "from sklearn.linear_model import" in line:
            import_src.insert(i, "from sklearn.neural_network import MLPClassifier\n")
            break

# Add new Markdown cell for Step 3: Model Comparison
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 🤖 Step 3 — Deep Learning Model Comparison\n",
        "We train 3 advanced models on our 4 computed signals:\n",
        "1. **LightGBM**: Gradient boosting trees.\n",
        "2. **Deep Neural Network (4 Layers)**: An MLP with architecture 128->64->32->16.\n",
        "3. **Wide Neural Network (2 Layers)**: An MLP with architecture 256->128."
    ],
    "id": "step3_dl_md"
})

# Add Code cell for training the models
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "# CELL 8 — Train and Compare 3 Models\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "FEATURES = ['med_isp', 'c_aas', 'med_eem', 'cdt']\n",
        "X = df_all[FEATURES].values\n",
        "y = df_all['label'].values\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.30, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_train_s = scaler.fit_transform(X_train)\n",
        "X_test_s = scaler.transform(X_test)\n",
        "\n",
        "models = {\n",
        "    'LightGBM': lgb.LGBMClassifier(\n",
        "        n_estimators=300, learning_rate=0.05, max_depth=5,\n",
        "        class_weight='balanced', random_state=42, verbose=-1\n",
        "    ),\n",
        "    'DNN (Deep)': MLPClassifier(\n",
        "        hidden_layer_sizes=(128, 64, 32, 16), activation='relu',\n",
        "        solver='adam', max_iter=1000, random_state=42, early_stopping=True\n",
        "    ),\n",
        "    'DNN (Wide)': MLPClassifier(\n",
        "        hidden_layer_sizes=(256, 128), activation='relu',\n",
        "        solver='adam', max_iter=1000, random_state=42, early_stopping=True\n",
        "    )\n",
        "}\n",
        "\n",
        "results = []\n",
        "trained_models = {}\n",
        "\n",
        "print('Training 3 Models on 4 CLINIGUARD signals...')\n",
        "for name, model in models.items():\n",
        "    print(f'  -> Training {name}...')\n",
        "    model.fit(X_train_s, y_train)\n",
        "    trained_models[name] = model\n",
        "    \n",
        "    y_prob = model.predict_proba(X_test_s)[:, 1]\n",
        "    y_pred = model.predict(X_test_s)\n",
        "    \n",
        "    auroc = roc_auc_score(y_test, y_prob)\n",
        "    ap = average_precision_score(y_test, y_prob)\n",
        "    f1 = f1_score(y_test, y_pred, zero_division=0)\n",
        "    \n",
        "    prec, rec, _ = precision_recall_curve(y_test, y_prob)\n",
        "    idx = np.argmin(np.abs(rec - 0.95))\n",
        "    p95r = prec[idx]\n",
        "    \n",
        "    results.append({\n",
        "        'Model': name,\n",
        "        'AUROC': auroc,\n",
        "        'Avg Precision': ap,\n",
        "        'F1-Score': f1,\n",
        "        'Prec@95%Recall': p95r\n",
        "    })\n",
        "\n",
        "df_res = pd.DataFrame(results)\n",
        "display(df_res.style.background_gradient(cmap='viridis'))"
    ],
    "id": "train_dl_models"
})

# Add Markdown for Visualization
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 📊 Step 4 — Visualize Comparison\n",
        "Let's visualize the AUROC and F1-Scores across all 3 models."
    ],
    "id": "step4_dl_md"
})

# Add Code for Visualization
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "# CELL 9 — Performance Bar Chart\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "df_melted = df_res.melt(id_vars='Model', value_vars=['AUROC', 'F1-Score', 'Avg Precision'], var_name='Metric', value_name='Score')\n",
        "\n",
        "sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette=['#4e79a7','#f28e2b','#e15759'], ax=ax, edgecolor='black')\n",
        "ax.set_ylim(0, 1.0)\n",
        "ax.set_title('Deep Learning Models vs LightGBM Performance', fontsize=14, fontweight='bold')\n",
        "ax.set_ylabel('Score')\n",
        "ax.set_xlabel('Metric')\n",
        "\n",
        "for p in ax.patches:\n",
        "    if p.get_height() > 0:\n",
        "        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),\n",
        "                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')\n",
        "\n",
        "plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Conclusion: LightGBM typically outperforms Deep Neural Networks on structured tabular features.')"
    ],
    "id": "viz_dl_comparison"
})

# Add Final Markdown
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## ✅ CLINIGUARD DL Comparison Complete\n",
        "You can run inference using the winning model just like before!"
    ],
    "id": "summary_dl_md"
})

nb['cells'] = new_cells

with open('CLINIGUARD_DL_Comparison.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Created CLINIGUARD_DL_Comparison.ipynb successfully!")
