"""
select_images.py — 把 benchmark 里用到的图片从图片库挑出来复制到单独目录

用法:
    python select_images.py [--gt benchmark.csv benchmark_wrinkles.csv]
                            [--img-dir CelebA-HQ-img] [--out benchmark_images] [--prune]

--gt   可传多个 benchmark 文件, 取并集
--prune 删除输出目录里 benchmark 不再需要的残留图片
"""

import argparse
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", nargs="+", default=["data/benchmark/benchmark.csv", "data/benchmark/benchmark_wrinkles.csv"])
    ap.add_argument("--img-dir", default="CelebA-HQ-img")
    ap.add_argument("--out", default="benchmark_images")
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()

    images = set()
    for gt in args.gt:
        images |= set(pd.read_csv(gt)["image_name"].astype(str))
    images = sorted(images)
    print(f"{len(args.gt)} 个 benchmark 去重后共 {len(images)} 张图片")

    os.makedirs(args.out, exist_ok=True)

    copied, missing = 0, []
    for name in images:
        src = os.path.join(args.img_dir, name)
        dst = os.path.join(args.out, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1

    print(f"已确保 {copied} 张在 {args.out}/")
    if missing:
        print(f"警告: {len(missing)} 张在 {args.img_dir} 中找不到, 例如: {missing[:5]}")

    if args.prune:
        need = set(images)
        stale = [f for f in os.listdir(args.out) if f not in need]
        for f in stale:
            os.remove(os.path.join(args.out, f))
        print(f"已清理残留旧图 {len(stale)} 张")


if __name__ == "__main__":
    main()
