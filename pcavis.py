import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import AllChem
import bitbirch.bitbirch as bb
from tqdm.auto import tqdm
import threading
from scipy.stats import gaussian_kde
import mplcursors
from PIL import Image, ImageTk
import io

class BitBirchPCAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BitBirch PCA Clustering Visualizer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.data = None
        self.X = None  # Feature matrix
        self.pca_result = None
        self.brc = None  # BitBirch clusterer
        self.cluster_assignments = None
        self.centroids = None
        self.centroid_pca = None
        self.current_view = 'overview'  # 'overview' or 'cluster'
        self.selected_cluster = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Load SMILES CSV", 
                  command=self.load_smiles_file).grid(row=0, column=0, padx=(0, 10))
        
        self.file_label = ttk.Label(control_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=(0, 20))
        
        # BitBirch parameters
        ttk.Label(control_frame, text="BB Threshold:").grid(row=0, column=2, padx=(0, 5))
        self.threshold_var = tk.StringVar(value="0.65")
        threshold_entry = ttk.Entry(control_frame, textvariable=self.threshold_var, width=8)
        threshold_entry.grid(row=0, column=3, padx=(0, 10))
        
        ttk.Label(control_frame, text="Branching Factor:").grid(row=0, column=4, padx=(0, 5))
        self.branching_var = tk.StringVar(value="50")
        branching_entry = ttk.Entry(control_frame, textvariable=self.branching_var, width=8)
        branching_entry.grid(row=0, column=5, padx=(0, 10))
        
        # Fingerprint parameters
        ttk.Label(control_frame, text="FP Radius:").grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        self.radius_var = tk.StringVar(value="2")
        radius_entry = ttk.Entry(control_frame, textvariable=self.radius_var, width=8)
        radius_entry.grid(row=1, column=1, padx=(0, 10), pady=(5, 0))
        
        ttk.Label(control_frame, text="FP Bits:").grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        self.nbits_var = tk.StringVar(value="2048")
        nbits_entry = ttk.Entry(control_frame, textvariable=self.nbits_var, width=8)
        nbits_entry.grid(row=1, column=3, padx=(0, 10), pady=(5, 0))
        
        # Action buttons
        self.process_btn = ttk.Button(control_frame, text="Process & Cluster", 
                                     command=self.start_processing, state="disabled")
        self.process_btn.grid(row=1, column=4, padx=(0, 10), pady=(5, 0))
        
        self.back_btn = ttk.Button(control_frame, text="← Back to Overview", 
                                  command=self.show_overview, state="disabled")
        self.back_btn.grid(row=1, column=5, pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Data info frame
        info_frame = ttk.LabelFrame(main_frame, text="Data Information", padding="5")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)
        
        self.info_text = tk.Text(info_frame, height=8, width=40)
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        plot_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind click event
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Clustering Results", padding="5")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        
        self.results_text = tk.Text(results_frame, height=12)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def load_smiles_file(self):
        file_path = filedialog.askopenfilename(
            title="Select SMILES CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Try to detect separator and column structure
                sample = pd.read_csv(file_path, nrows=5)
                
                if len(sample.columns) == 1:
                    # Single column, might be space-separated
                    self.data = pd.read_csv(file_path, sep=" ", names=["SMILES", "Name"])
                elif 'SMILES' in sample.columns or 'smiles' in sample.columns:
                    # Already has SMILES column
                    self.data = pd.read_csv(file_path)
                    if 'smiles' in self.data.columns:
                        self.data.rename(columns={'smiles': 'SMILES'}, inplace=True)
                else:
                    # Assume first column is SMILES
                    self.data = pd.read_csv(file_path)
                    cols = list(self.data.columns)
                    self.data.rename(columns={cols[0]: 'SMILES'}, inplace=True)
                    if len(cols) > 1:
                        self.data.rename(columns={cols[1]: 'Name'}, inplace=True)
                
                self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                self.process_btn.config(state="normal")
                
                # Display data info
                self.display_data_info()
                
                messagebox.showinfo("Success", f"SMILES data loaded successfully!\nShape: {self.data.shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def display_data_info(self):
        if self.data is None:
            return
            
        info = []
        info.append(f"Dataset Shape: {self.data.shape}")
        info.append(f"Columns: {list(self.data.columns)}")
        
        if 'SMILES' in self.data.columns:
            info.append(f"\nSMILES Examples:")
            for i, smiles in enumerate(self.data['SMILES'].head(5)):
                info.append(f"  {i+1}: {smiles}")
        
        if hasattr(self, 'X') and self.X is not None:
            info.append(f"\nFingerprint Matrix: {self.X.shape}")
        
        if hasattr(self, 'cluster_assignments') and self.cluster_assignments is not None:
            unique_clusters = np.unique(self.cluster_assignments[self.cluster_assignments >= 0])
            info.append(f"\nNumber of Clusters: {len(unique_clusters)}")
            info.append(f"Noise Points: {np.sum(self.cluster_assignments == -1)}")
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info))
    
    def mol2fp(self, mol):
        """Convert molecule to Morgan fingerprint"""
        radius = int(self.radius_var.get())
        nbits = int(self.nbits_var.get())
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    
    def start_processing(self):
        """Start processing in a separate thread to avoid freezing GUI"""
        self.progress.start()
        self.process_btn.config(state="disabled")
        
        thread = threading.Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
    
    def process_data(self):
        """Process SMILES data and perform clustering"""
        try:
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Converting SMILES to fingerprints...\n"))
            
            # Convert SMILES to molecules
            tqdm.pandas(desc="Converting SMILES")
            self.data['mol'] = self.data.SMILES.progress_apply(Chem.MolFromSmiles)
            
            # Remove invalid molecules
            valid_mols = self.data['mol'].notna()
            if not valid_mols.all():
                invalid_count = (~valid_mols).sum()
                self.root.after(0, lambda: self.results_text.insert(tk.END, f"Removed {invalid_count} invalid SMILES\n"))
                self.data = self.data[valid_mols].reset_index(drop=True)
            
            # Convert to fingerprints
            tqdm.pandas(desc="Generating fingerprints")
            self.data['fp'] = self.data.mol.progress_apply(self.mol2fp)
            
            # Convert to numpy array
            self.X = np.stack(self.data.fp.apply(lambda x: np.array(list(x.ToBitString()), dtype=np.uint8)))
            
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Generated fingerprint matrix: {self.X.shape}\n"))
            
            # Perform BitBirch clustering
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Performing BitBirch clustering...\n"))
            
            bb.set_merge('diameter')
            threshold = float(self.threshold_var.get())
            branching_factor = int(self.branching_var.get())
            
            self.brc = bb.BitBirch(branching_factor=branching_factor, threshold=threshold)
            self.brc.fit(self.X)
            
            # Get cluster indices
            clust_indices = self.brc.get_cluster_mol_ids()
            
            # Create cluster assignments array
            self.cluster_assignments = np.ones(self.X.shape[0], dtype='int64') * -1
            
            for label, cluster in enumerate(clust_indices):
                self.cluster_assignments[cluster] = label
            
            self.data['cluster'] = self.cluster_assignments
            num_clusters = len(clust_indices)
            
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Number of clusters: {num_clusters}\n"))
            
            # Get centroids and compute PCA
            self.centroids = self.brc.get_centroids()
            self.centroid_pca = self.compute_pca(self.centroids)
            
            if not self.centroid_pca.empty:
                self.centroid_pca['cluster'] = range(num_clusters)
                self.centroid_pca['size'] = self.data['cluster'].value_counts().sort_index().values
                self.centroid_pca['hover'] = self.centroid_pca['cluster'].apply(lambda x: f"Cluster {x}")
            
            # Update GUI
            self.root.after(0, self.finish_processing)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
    
    def finish_processing(self):
        """Finish processing and update GUI"""
        self.progress.stop()
        self.process_btn.config(state="normal")
        self.display_data_info()
        self.show_overview()
        self.display_clustering_results()
    
    def compute_pca(self, data):
        """Compute PCA for visualization"""
        if len(data) < 2:
            return pd.DataFrame()
        
        # Normalize data
        #norms = np.linalg.norm(data, axis=1, keepdims=True)
        #norms[norms == 0] = 1
        #data_normalized = data / norms
        
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(data)
        
        return pd.DataFrame(transformed, columns=['PC1', 'PC2'])
    
    def show_overview(self):
        """Show overview plot with cluster centroids (minimum 10 molecules)"""
        if self.centroid_pca is None or self.centroid_pca.empty:
            return
        
        self.current_view = 'overview'
        self.selected_cluster = None
        self.back_btn.config(state="disabled")
        
        # Filter clusters with minimum 10 molecules
        min_cluster_size = 10
        large_clusters = self.centroid_pca[self.centroid_pca['size'] >= min_cluster_size].copy()
        
        if large_clusters.empty:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No clusters with ≥ {min_cluster_size} molecules found.\nTry lowering the BitBirch threshold.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('BitBirch Cluster Overview')
            self.fig.tight_layout()
            self.canvas.draw()
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create color map
        n_clusters = len(large_clusters)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # Plot cluster centroids with improved styling
        sizes = large_clusters['size'].values
        # Scale sizes more reasonably (50-500 range)
        scaled_sizes = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) * 450
        
        scatter = ax.scatter(large_clusters['PC1'], large_clusters['PC2'], 
                           s=scaled_sizes,
                           c=colors,
                           alpha=0.8, 
                           edgecolors='white', 
                           linewidths=2)
        
        # Add cluster labels with better positioning
        for i, (idx, row) in enumerate(large_clusters.iterrows()):
            ax.annotate(f"C{row['cluster']}\n({row['size']})", 
                       (row['PC1'], row['PC2']), 
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       weight='bold')
        
        # Styling improvements
        ax.set_xlabel('Principal Component 1', fontsize=12, weight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=12, weight='bold')
        ax.set_title(f'BitBirch Cluster Overview (≥{min_cluster_size} molecules)\nClick on clusters to explore', 
                    fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#fafafa')
        
        # Add legend for cluster sizes
        size_legend_elements = []
        size_ranges = [(10, 50), (50, 100), (100, float('inf'))]
        size_labels = ['10-50', '50-100', '100+']
        legend_sizes = [100, 200, 300]
        
        for i, (size_range, label, legend_size) in enumerate(zip(size_ranges, size_labels, legend_sizes)):
            if any((sizes >= size_range[0]) & (sizes < size_range[1])):
                size_legend_elements.append(plt.scatter([], [], s=legend_size, c='gray', alpha=0.6, 
                                                       edgecolors='white', linewidths=1, label=f'{label} molecules'))
        
        if size_legend_elements:
            ax.legend(handles=size_legend_elements, loc='upper right', title='Cluster Size', 
                     frameon=True, fancybox=True, shadow=True)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_cluster_detail(self, cluster_id):
            
        """Show detailed view of a specific cluster"""
        if self.data is None or self.X is None:
            return

        # Get molecules in this cluster
        cluster_mask = self.data['cluster'] == cluster_id
        cluster_data = self.X[cluster_mask]
        cluster_df = self.data[cluster_mask].reset_index(drop=True)

        if len(cluster_data) < 10:
            messagebox.showinfo("Info", f"Cluster {cluster_id} has only {len(cluster_data)} molecules. Minimum 10 required for detailed view.")
            return

        if len(cluster_data) < 2:
            return

        # Compute PCA for this cluster
        cluster_pca = self.compute_pca(cluster_data)
        if cluster_pca.empty:
            return

        self.current_view = 'cluster'
        self.selected_cluster = cluster_id
        self.back_btn.config(state="normal")

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Create density-based coloring
        xy = np.vstack([cluster_pca['PC1'], cluster_pca['PC2']])
        density = gaussian_kde(xy)(xy)

        scatter = ax.scatter(cluster_pca['PC1'], cluster_pca['PC2'],
                             c=density, s=40, alpha=0.7,
                             cmap='viridis', edgecolors='white', linewidths=0.5)

        # --- Add interactive hover popup for molecules ---
        mols = cluster_df['mol'].tolist()
        smiles_list = cluster_df['SMILES'].tolist()

        import mplcursors
        from rdkit.Chem import Draw
        from PIL import Image, ImageTk
        import io
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_hover(sel):
          """Show molecule image above the hovered point"""
          idx = sel.index
          mol = mols[idx]
      
          if mol is not None:
              # Generate RDKit image
              img = Draw.MolToImage(mol, size=(150, 150))
              bio = io.BytesIO()
              img.save(bio, format="PNG")
              bio.seek(0)
      
              arr_img = plt.imread(bio, format="PNG")
              imagebox = OffsetImage(arr_img, zoom=0.6)
      
              # Place image above the point
              ab = AnnotationBbox(
                  imagebox,
                  (sel.target[0], sel.target[1]),   # point coords
                  xybox=(0, 50),                   # offset above point
                  xycoords='data',
                  boxcoords="offset points",
                  frameon=True,
                  pad=0.3
              )
      
              # Hide default text
              sel.annotation.set_visible(False)
      
              # Add to axis
              sel.artist.axes.add_artist(ab)
              sel.extras.append(ab)  # ensures cleanup when hover leaves

        # ---- rest of your plotting code (colorbar, labels, etc.) ----
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Density', rotation=270, labelpad=15)

        ax.set_xlabel('Principal Component 1', fontsize=12, weight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=12, weight='bold')
        ax.set_title(f'Cluster {cluster_id} Detail View\n{len(cluster_data)} molecules',
                     fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#fafafa')

        stats_text = f'Molecules: {len(cluster_data)}\nSpread: {cluster_pca.std().mean():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()

        # Update cluster details in the right panel
        self.display_cluster_details(cluster_id, cluster_df)

    
    def on_plot_click(self, event):
        """Handle plot click events"""
        if event.inaxes is None or self.current_view != 'overview':
            return
        
        if self.centroid_pca is None or self.centroid_pca.empty:
            return
        
        # Only consider clusters with >= 10 molecules
        min_cluster_size = 10
        large_clusters = self.centroid_pca[self.centroid_pca['size'] >= min_cluster_size].copy()
        
        if large_clusters.empty:
            return
        
        # Find closest centroid to click
        click_point = np.array([event.xdata, event.ydata])
        centroids_points = large_clusters[['PC1', 'PC2']].values
        
        distances = np.linalg.norm(centroids_points - click_point, axis=1)
        closest_idx = np.argmin(distances)
        
        # If click is reasonably close, show cluster detail
        if distances[closest_idx] < 1.0:  # Increased threshold for better usability
            cluster_id = large_clusters.iloc[closest_idx]['cluster']
            self.show_cluster_detail(cluster_id)
    
    def display_clustering_results(self):
        """Display clustering results"""
        if self.data is None or self.cluster_assignments is None:
            return
        
        results = []
        results.append("BITBIRCH CLUSTERING RESULTS")
        results.append("=" * 40)
        
        unique_clusters = np.unique(self.cluster_assignments[self.cluster_assignments >= 0])
        min_cluster_size = 10
        large_clusters = sum(1 for cluster_id in unique_clusters 
                           if sum(self.cluster_assignments == cluster_id) >= min_cluster_size)
        
        results.append(f"\nTotal clusters found: {len(unique_clusters)}")
        results.append(f"Large clusters (≥{min_cluster_size} molecules): {large_clusters}")
        results.append(f"Noise points: {np.sum(self.cluster_assignments == -1)}")
        
        results.append(f"\nLarge cluster details:")
        cluster_sizes = pd.Series(self.cluster_assignments).value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            if cluster_id >= 0 and size >= min_cluster_size:
                percentage = (size / len(self.data)) * 100
                results.append(f"  Cluster {cluster_id}: {size} molecules ({percentage:.1f}%)")
        
        small_clusters = [size for cluster_id, size in cluster_sizes.items() 
                         if cluster_id >= 0 and size < min_cluster_size]
        if small_clusters:
            results.append(f"\nSmall clusters (<{min_cluster_size} molecules): {len(small_clusters)} clusters")
            results.append(f"  Total molecules in small clusters: {sum(small_clusters)}")
        
        results.append(f"\nParameters used:")
        results.append(f"  Threshold: {self.threshold_var.get()}")
        results.append(f"  Branching factor: {self.branching_var.get()}")
        results.append(f"  Fingerprint radius: {self.radius_var.get()}")
        results.append(f"  Fingerprint bits: {self.nbits_var.get()}")
        
        results.append(f"\n📍 Click on cluster centroids in the overview plot to explore!")
        results.append(f"💡 Only clusters with ≥{min_cluster_size} molecules are shown and clickable.")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "\n".join(results))
    
    def display_cluster_details(self, cluster_id, cluster_df):
        """Display details for a specific cluster"""
        results = []
        results.append(f"CLUSTER {cluster_id} DETAILS")
        results.append("=" * 30)
        
        results.append(f"\nCluster size: {len(cluster_df)} molecules")
        
        if 'Name' in cluster_df.columns:
            results.append(f"\nSample molecules:")
            for i, (idx, row) in enumerate(cluster_df.head(10).iterrows()):
                name = row.get('Name', f'Molecule_{idx}')
                smiles = row['SMILES']
                results.append(f"  {i+1}. {name}: {smiles}")
        else:
            results.append(f"\nSample SMILES:")
            for i, smiles in enumerate(cluster_df['SMILES'].head(10)):
                results.append(f"  {i+1}. {smiles}")
        
        if len(cluster_df) > 10:
            results.append(f"  ... and {len(cluster_df) - 10} more molecules")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "\n".join(results))

def main():
    root = tk.Tk()
    app = BitBirchPCAGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()