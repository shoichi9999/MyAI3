#!/usr/bin/env python3
"""
POG予測システム - エントリーポイント

使い方:
  # 全自動実行（データ収集→予測）
  python run.py

  # 2024年生まれの馬を対象に、テスト用に20頭だけ取得して実行
  python run.py --year 2024 --max-horses 20

  # データ収集のみ
  python run.py --mode collect --year 2024

  # 予測のみ（収集済みデータ使用）
  python run.py --mode predict --year 2024 --top-n 10
"""

from src.predictor import main

if __name__ == "__main__":
    main()
