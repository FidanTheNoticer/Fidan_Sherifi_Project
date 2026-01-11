"""
Application graphique pour prédire la santé financière - VERSION SIMPLE
Interface redimensionnable sans dark mode.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class FinancialHealthApp:
    """Interface graphique simple et redimensionnable."""

    def __init__(self, root):
        self.root = root
        self.root.title("🏦 Classification Santé Financière Suisse")
        self.root.geometry("1300x700")
        self.root.minsize(1200, 650)

        # Couleurs simples
        self.colors = {
            'bg_primary': '#f5f7fa',
            'bg_card': '#ffffff',
            'accent': '#2563eb',
            'accent_hover': '#1d4ed8',
            'text_primary': '#1f2937',
            'text_secondary': '#6b7280',
            'border': '#e5e7eb',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }

        self.root.configure(bg=self.colors['bg_primary'])

        # Charger le modèle
        try:
            self.model = joblib.load('models/best_classifier.pkl')
            self.encoders = joblib.load('models/encoders.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.class_names = joblib.load('models/class_names.pkl')
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle: {e}")
            root.destroy()
            return

        # Créer l'interface
        self.create_widgets()

    def create_widgets(self):
        """Créer tous les widgets."""

        # Container principal
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # En-tête simple
        header_frame = tk.Frame(main_container, bg=self.colors['bg_card'], 
                               relief='flat', bd=1, 
                               highlightbackground=self.colors['border'],
                               highlightthickness=1)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        title = tk.Label(header_frame, text="🏦 Analyse de Santé Financière", 
                        bg=self.colors['bg_card'],
                        fg=self.colors['text_primary'],
                        font=('Segoe UI', 18, 'bold'))
        title.pack(pady=(12, 3))

        subtitle = tk.Label(header_frame, 
                           text="Évaluation basée sur 11 indicateurs financiers clés", 
                           bg=self.colors['bg_card'],
                           fg=self.colors['text_secondary'],
                           font=('Segoe UI', 9, 'italic'))
        subtitle.pack(pady=(0, 12))

        # Frame principal avec 2 colonnes
        content_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Colonne gauche - Formulaire (largeur fixe réduite)
        left_frame = tk.Frame(content_frame, bg=self.colors['bg_card'],
                             relief='flat', bd=1, 
                             highlightbackground=self.colors['border'],
                             highlightthickness=1, width=260)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_frame.pack_propagate(False)

        # Titre du formulaire
        form_title = tk.Label(left_frame, text="📝 Vos Informations", 
                             bg=self.colors['bg_card'],
                             fg=self.colors['text_primary'],
                             font=('Segoe UI', 11, 'bold'))
        form_title.pack(pady=(12, 8), padx=12)

        # Canvas pour les inputs avec scrollbar
        canvas_container = tk.Frame(left_frame, bg=self.colors['bg_card'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=(12, 0))

        input_canvas = tk.Canvas(canvas_container, bg=self.colors['bg_card'],
                                highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_container, orient="vertical", 
                                command=input_canvas.yview)
        input_container = tk.Frame(input_canvas, bg=self.colors['bg_card'])

        input_container.bind(
            "<Configure>",
            lambda e: input_canvas.configure(scrollregion=input_canvas.bbox("all"))
        )

        input_canvas.create_window((0, 0), window=input_container, anchor="nw", width=230)
        input_canvas.configure(yscrollcommand=scrollbar.set)

        input_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Variables
        self.age_var = tk.IntVar(value=35)
        self.canton_var = tk.StringVar()
        self.situation_var = tk.StringVar()
        self.enfants_var = tk.IntVar(value=0)
        self.occupation_var = tk.IntVar(value=100)
        self.salaire_var = tk.IntVar(value=7000)
        self.loyer_var = tk.IntVar(value=2000)
        self.vitales_var = tk.IntVar(value=1500)
        self.loisirs_var = tk.IntVar(value=800)
        self.credit_var = tk.StringVar()
        self.credit_montant_var = tk.IntVar(value=0)

        # Créer les inputs
        self.create_input_fields(input_container)

        # Bouton Analyser (fixé en bas)
        btn_frame = tk.Frame(left_frame, bg=self.colors['bg_card'])
        btn_frame.pack(side=tk.BOTTOM, pady=(8, 12), padx=12)

        analyze_btn = tk.Button(btn_frame, text="🔍 Analyser ma situation",
                               command=self.predict,
                               bg=self.colors['accent'], fg='white',
                               font=('Segoe UI', 10, 'bold'),
                               relief='flat', bd=0,
                               padx=25, pady=10,
                               cursor='hand2',
                               activebackground=self.colors['accent_hover'],
                               activeforeground='white')
        analyze_btn.pack()

        # Colonne droite - Résultats (expansible)
        right_frame = tk.Frame(content_frame, bg=self.colors['bg_primary'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas pour résultats avec scrollbar
        self.results_canvas = tk.Canvas(right_frame, bg=self.colors['bg_primary'],
                                       highlightthickness=0)
        results_scrollbar = tk.Scrollbar(right_frame, orient="vertical", 
                                        command=self.results_canvas.yview)
        self.results_container = tk.Frame(self.results_canvas, bg=self.colors['bg_primary'])

        self.results_container.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )

        self.results_canvas.create_window((0, 0), window=self.results_container, 
                                         anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Message initial
        self.show_welcome_message()

    def _on_mousewheel(self, event):
        """Gérer le scroll avec la molette."""
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_input_fields(self, parent):
        """Créer les champs de saisie compacts."""

        fields = [
            ("👤 Âge", self.age_var, 25, 65, None),
            ("📍 Canton", self.canton_var, None, None, list(self.encoders['canton'].classes_)),
            ("💑 Situation", self.situation_var, None, None, list(self.encoders['situation_maritale'].classes_)),
            ("👶 Enfants", self.enfants_var, 0, 5, None),
            ("💼 Occupation (%)", self.occupation_var, 50, 100, None),
            ("💰 Salaire (CHF)", self.salaire_var, 2000, 15000, None),
            ("🏠 Loyer (CHF)", self.loyer_var, 500, 5000, None),
            ("🛒 Dépenses vitales (CHF)", self.vitales_var, 300, 4000, None),
            ("🎉 Loisirs (CHF)", self.loisirs_var, 0, 3000, None),
            ("💳 Crédit", self.credit_var, None, None, list(self.encoders['a_credit'].classes_)),
            ("💵 Montant crédit (CHF)", self.credit_montant_var, 0, 4000, None)
        ]

        for i, (label, var, min_val, max_val, values) in enumerate(fields):
            field_frame = tk.Frame(parent, bg=self.colors['bg_card'])
            field_frame.pack(fill=tk.X, pady=3)

            # Label
            lbl = tk.Label(field_frame, text=label, 
                          bg=self.colors['bg_card'],
                          fg=self.colors['text_primary'],
                          font=('Segoe UI', 9, 'bold'),
                          anchor='w')
            lbl.pack(fill=tk.X)

            # Widget
            if values is not None:
                widget = ttk.Combobox(field_frame, textvariable=var, 
                                     values=values, state='readonly',
                                     font=('Segoe UI', 9), height=8)
                widget.pack(fill=tk.X, pady=(2, 0))
                if i == 1:
                    widget.set(values[0])
                elif i == 2:
                    widget.set(values[0])
                elif i == 9:
                    widget.set(values[0])
                    widget.bind('<<ComboboxSelected>>', self.toggle_credit_amount)
            else:
                widget = tk.Spinbox(field_frame, textvariable=var,
                                   from_=min_val, to=max_val,
                                   font=('Segoe UI', 9),
                                   relief='solid', bd=1,
                                   increment=100 if min_val >= 100 else 10)
                widget.pack(fill=tk.X, pady=(2, 0))

                if i == 10:
                    self.credit_spinbox = widget
                    widget.config(state='disabled')

    def toggle_credit_amount(self, event=None):
        """Activer/désactiver le montant de crédit."""
        if self.credit_var.get() == 'oui':
            self.credit_spinbox.config(state='normal')
        else:
            self.credit_montant_var.set(0)
            self.credit_spinbox.config(state='disabled')

    def show_welcome_message(self):
        """Afficher le message de bienvenue."""
        for widget in self.results_container.winfo_children():
            widget.destroy()

        welcome_frame = tk.Frame(self.results_container, bg=self.colors['bg_card'],
                                relief='flat', bd=1, 
                                highlightbackground=self.colors['border'],
                                highlightthickness=1)
        welcome_frame.pack(fill=tk.BOTH, expand=True)

        # Centrer le contenu
        center_frame = tk.Frame(welcome_frame, bg=self.colors['bg_card'])
        center_frame.place(relx=0.5, rely=0.5, anchor='center')

        icon = tk.Label(center_frame, text="📊", 
                       bg=self.colors['bg_card'],
                       font=('Segoe UI', 70))
        icon.pack(pady=(0, 15))

        msg = tk.Label(center_frame, 
                      text="Complétez le formulaire\net cliquez sur 'Analyser'",
                      bg=self.colors['bg_card'],
                      fg=self.colors['text_secondary'],
                      font=('Segoe UI', 13),
                      justify=tk.CENTER)
        msg.pack()

    def predict(self):
        """Effectuer la prédiction."""
        try:
            data = {
                'age': self.age_var.get(),
                'canton': self.canton_var.get(),
                'situation_maritale': self.situation_var.get(),
                'nombre_enfants': self.enfants_var.get(),
                'taux_occupation': self.occupation_var.get(),
                'salaire_mensuel': self.salaire_var.get(),
                'loyer_mensuel': self.loyer_var.get(),
                'depenses_vitales': self.vitales_var.get(),
                'depenses_loisirs': self.loisirs_var.get(),
                'a_credit': self.credit_var.get(),
                'montant_credit_mensuel': self.credit_montant_var.get()
            }

            df = pd.DataFrame([data])
            df['canton_encoded'] = self.encoders['canton'].transform([data['canton']])[0]
            df['situation_encoded'] = self.encoders['situation_maritale'].transform([data['situation_maritale']])[0]
            df['credit_encoded'] = self.encoders['a_credit'].transform([data['a_credit']])[0]

            X = df[self.feature_names]
            prediction = self.model.predict(X)[0]
            probas = self.model.predict_proba(X)[0]

            self.display_results(prediction, probas, data)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur: {e}")

    def display_results(self, prediction, probas, data):
        """Afficher les résultats."""

        for widget in self.results_container.winfo_children():
            widget.destroy()

        # Calculer métriques
        ratio_loyer = (data['loyer_mensuel'] / data['salaire_mensuel'] * 100)
        depenses_totales = (data['loyer_mensuel'] + data['depenses_vitales'] + 
                           data['depenses_loisirs'] + data['montant_credit_mensuel'])
        epargne = data['salaire_mensuel'] - depenses_totales
        taux_epargne = (epargne / data['salaire_mensuel'] * 100) if data['salaire_mensuel'] > 0 else 0

        # Mapping
        class_labels = {
            'très_mauvaise': '😰 Très Préoccupante',
            'mauvaise': '😟 Préoccupante',
            'moyenne': '😐 Fragile',
            'bonne': '😊 Saine',
            'très_bonne': '🎉 Excellente'
        }

        class_colors_bg = {
            'très_mauvaise': '#fee2e2',
            'mauvaise': '#fed7aa',
            'moyenne': '#fef3c7',
            'bonne': '#d9f99d',
            'très_bonne': '#bbf7d0'
        }

        class_colors_text = {
            'très_mauvaise': '#991b1b',
            'mauvaise': '#9a3412',
            'moyenne': '#854d0e',
            'bonne': '#365314',
            'très_bonne': '#14532d'
        }

        # Card résultat
        result_card = tk.Frame(self.results_container, 
                              bg=class_colors_bg[prediction],
                              relief='flat', bd=1,
                              highlightbackground=class_colors_text[prediction],
                              highlightthickness=2)
        result_card.pack(fill=tk.X, pady=(0, 8), padx=1)

        tk.Label(result_card, text="Votre situation financière est :",
                bg=class_colors_bg[prediction],
                fg=class_colors_text[prediction],
                font=('Segoe UI', 10)).pack(pady=(15, 3))

        tk.Label(result_card, text=class_labels[prediction],
                bg=class_colors_bg[prediction],
                fg=class_colors_text[prediction],
                font=('Segoe UI', 24, 'bold')).pack(pady=(0, 15))

        # Métriques
        metrics_frame = tk.Frame(self.results_container, bg=self.colors['bg_card'],
                                relief='flat', bd=1,
                                highlightbackground=self.colors['border'],
                                highlightthickness=1)
        metrics_frame.pack(fill=tk.X, pady=(0, 8), padx=1)

        tk.Label(metrics_frame, text="📊 Indicateurs Clés",
                bg=self.colors['bg_card'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 11, 'bold')).pack(pady=(10, 5), padx=15, anchor='w')

        metrics = [
            ("Ratio Loyer/Salaire", f"{ratio_loyer:.1f}%", ratio_loyer > 35),
            ("Épargne mensuelle", f"{epargne:.0f} CHF ({taux_epargne:.1f}%)", epargne < 0),
            ("Dépenses totales", f"{depenses_totales:.0f} CHF", False)
        ]

        for label, value, is_warning in metrics:
            row = tk.Frame(metrics_frame, bg=self.colors['bg_card'])
            row.pack(fill=tk.X, padx=15, pady=3)

            tk.Label(row, text=label,
                    bg=self.colors['bg_card'],
                    fg=self.colors['text_secondary'],
                    font=('Segoe UI', 9)).pack(side=tk.LEFT)

            tk.Label(row, text=value,
                    bg=self.colors['bg_card'],
                    fg=self.colors['danger'] if is_warning else self.colors['success'],
                    font=('Segoe UI', 9, 'bold')).pack(side=tk.RIGHT)

        tk.Frame(metrics_frame, bg=self.colors['bg_card'], height=10).pack()

        # Recommandations
        self.display_recommendations(data, ratio_loyer, epargne)

        # # Graphique
        # self.display_chart(probas)

        # Scroll en haut
        self.results_canvas.yview_moveto(0)

    def display_recommendations(self, data, ratio_loyer, epargne):
        """Afficher les recommandations."""

        reco_frame = tk.Frame(self.results_container, bg=self.colors['bg_card'],
                             relief='flat', bd=1,
                             highlightbackground=self.colors['border'],
                             highlightthickness=1)
        reco_frame.pack(fill=tk.X, pady=(0, 8), padx=1)

        tk.Label(reco_frame, text="💡 Recommandations Personnalisées",
                bg=self.colors['bg_card'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 11, 'bold')).pack(pady=(10, 5), padx=15, anchor='w')

        recommandations = []

        if ratio_loyer > 35:
            loyer_ideal = data['salaire_mensuel'] * 0.30
            recommandations.append(
                ("🏠", f"Votre loyer représente {ratio_loyer:.0f}% de vos revenus",
                 f"Objectif: réduire à {loyer_ideal:.0f} CHF (30%)")
            )

        ratio_loisirs = (data['depenses_loisirs'] / data['salaire_mensuel'] * 100)
        if ratio_loisirs > 15:
            economie = data['depenses_loisirs'] * 0.3
            recommandations.append(
                ("🎯", f"Dépenses de loisirs élevées ({ratio_loisirs:.0f}%)",
                 f"Économie potentielle: {economie:.0f} CHF/mois")
            )

        if data['a_credit'] == 'oui' and data['montant_credit_mensuel'] > 0:
            ratio_credit = (data['montant_credit_mensuel'] / data['salaire_mensuel'] * 100)
            if ratio_credit > 20:
                recommandations.append(
                    ("💳", f"Charges de crédit importantes ({ratio_credit:.0f}%)",
                     "Envisagez de renégocier les conditions")
                )

        if epargne < 0:
            recommandations.append(
                ("⚠️", "Situation de déficit budgétaire",
                 f"Réduire les dépenses de {abs(epargne):.0f} CHF/mois minimum")
            )

        if not recommandations:
            tk.Label(reco_frame,
                    text="✅ Excellente gestion financière ! Continuez ainsi.",
                    bg=self.colors['bg_card'],
                    fg=self.colors['success'],
                    font=('Segoe UI', 10, 'bold')).pack(pady=15, padx=15)
        else:
            for icon, title, detail in recommandations[:3]:
                item = tk.Frame(reco_frame, bg='#f9fafb', relief='flat')
                item.pack(fill=tk.X, padx=15, pady=5)

                tk.Label(item, text=icon, bg='#f9fafb',
                        font=('Segoe UI', 14)).pack(side=tk.LEFT, padx=10, pady=8)

                text_frame = tk.Frame(item, bg='#f9fafb')
                text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)

                tk.Label(text_frame, text=title, bg='#f9fafb',
                        fg=self.colors['text_primary'],
                        font=('Segoe UI', 9, 'bold'),
                        anchor='w').pack(fill=tk.X)

                tk.Label(text_frame, text=detail, bg='#f9fafb',
                        fg=self.colors['text_secondary'],
                        font=('Segoe UI', 8),
                        anchor='w').pack(fill=tk.X, pady=(2, 0))

        tk.Frame(reco_frame, bg=self.colors['bg_card'], height=10).pack()

    def display_chart(self, probas):
        """Afficher le graphique."""

        chart_frame = tk.Frame(self.results_container, bg=self.colors['bg_card'],
                              relief='flat', bd=1,
                              highlightbackground=self.colors['border'],
                              highlightthickness=1)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=(0, 8))

        tk.Label(chart_frame, text="📈 Distribution des Probabilités",
                bg=self.colors['bg_card'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 11, 'bold')).pack(pady=(10, 5), padx=15, anchor='w')

        fig = Figure(figsize=(8, 3.2), facecolor=self.colors['bg_card'])
        ax = fig.add_subplot(111)

        colors = ['#ef4444', '#f59e0b', '#fbbf24', '#84cc16', '#22c55e']
        labels = ['Très\nMauvaise', 'Mauvaise', 'Moyenne', 'Bonne', 'Très\nBonne']

        bars = ax.barh(labels, probas * 100, color=colors, alpha=0.8)
        ax.set_xlabel('Probabilité (%)', fontsize=9, color=self.colors['text_primary'])
        ax.set_xlim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['border'])
        ax.spines['bottom'].set_color(self.colors['border'])
        ax.tick_params(colors=self.colors['text_secondary'], labelsize=8)
        fig.patch.set_facecolor(self.colors['bg_card'])
        ax.set_facecolor(self.colors['bg_card'])

        for bar, proba in zip(bars, probas):
            width = bar.get_width()
            if width > 5:
                ax.text(width - 2, bar.get_y() + bar.get_height()/2,
                       f'{proba*100:.1f}%',
                       ha='right', va='center', 
                       color='white', fontweight='bold', fontsize=8)

        fig.tight_layout(pad=1)

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=(0, 10), padx=15, fill=tk.BOTH, expand=True)


if __name__ == '__main__':
    root = tk.Tk()
    app = FinancialHealthApp(root)
    root.mainloop()
