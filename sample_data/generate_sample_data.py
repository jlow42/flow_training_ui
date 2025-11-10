import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    rng = np.random.default_rng(42)
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        size = 600 + idx * 100
        df = pd.DataFrame(
            {
                "FSC_A": rng.normal(110, 18, size),
                "SSC_A": rng.normal(65, 14, size),
                "Marker_CD3": rng.normal(5.5, 1.2, size),
                "Marker_CD19": rng.normal(2.1, 0.6, size),
                "Marker_CD56": rng.normal(3.0, 0.8, size),
                "Viability": rng.uniform(0.8, 1.0, size),
            }
        )
        labels = rng.choice(
            ["T-cell", "B-cell", "NK-cell"],
            size=size,
            p=[0.45, 0.35, 0.2],
        )
        df["CellType"] = labels
        df.to_csv(output_dir / f"demo_flow_{idx + 1}.csv", index=False)


if __name__ == "__main__":
    main()
