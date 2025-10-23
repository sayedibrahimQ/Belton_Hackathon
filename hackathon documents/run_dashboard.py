#!/usr/bin/env python3
"""
Dashboard runner for the Robin Logistics Environment.
 
Launch with: python run_dashboard.py
"""
 
import os
import sys
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
from robin_logistics import LogisticsEnvironment
from solver import my_solver
 
 
def main():
    env = LogisticsEnvironment()
    env.set_solver(my_solver)
    env.launch_dashboard()
 
 
if __name__ == "__main__":
    main()