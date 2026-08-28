from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_spline_tuning import run_spline_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一样条 Ridge 边界邻域实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_spline_tuning(
        ROOT,
        overwrite=args.overwrite,
        knots_values=(3, 4),
        degree_values=(2,),
        alpha_values=(0.3, 1.0, 3.0, 10.0),
        config_filename="task1_spline_refinement.toml",
        output_prefix="task1_non_sleep_spline_refinement",
        experiment_name="non_sleep_spline_refinement",
        control_model="spline_k4_d2_a3p0",
        control_oof_r2=0.7905766359789776,
    )
    print("任务一样条 Ridge 边界邻域实验完成。")
    for path in written:
        print(f"- {path}")
