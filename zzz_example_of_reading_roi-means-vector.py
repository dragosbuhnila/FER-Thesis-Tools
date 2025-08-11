import os
import numpy as np


if __name__ == "__main__":
    for roi_means_vector in os.listdir("zzz_output_means_vectors"):
        roi_means_vector_path = os.path.join("zzz_output_means_vectors", roi_means_vector)
        roi_means_vector_data = np.load(roi_means_vector_path, allow_pickle=True)

        # Mostra il primo file che trovi nella directory dei mean vectors
        print("======================================")
        print("Il 'vettore' appare come: (nota che è un dizionario di dizionario, non propriamente un vettore)")
        print(roi_means_vector_data)

        print("======================================")
        print("Il 'vettore' con un output formattato più bellamente è:")
        for ROI, stat in roi_means_vector_data.item().items():
            print(f"{ROI}: {stat}")
        print("======================================")
        break