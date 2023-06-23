from enum import Enum
from itertools import product
from typing import List, Tuple, Dict
import sys
from hitman_referee import HC, HitmanReferee, complete_map_example
from pprint import pprint

import subprocess
from itertools import combinations





Grid = List[List[int]]
Variable = int
Literal = int
Clause = List[Literal]
Map = List[List[Literal]]
Coord = Tuple[int, int]
PropositionnalVariable = int
ClauseBase = List[Clause]

def grid_to_coords_dict(grid: Grid) -> dict:
    """
    :param grid: grid of the level
    :return: dict containing position of each element
    """
    coords = {
        "cells": [],
        "empty": [],
        "hero": [],
        "guards": [],
        "targets": [],
        "walls": [],
        "costumes": [],
        "ropes": [],
        "nothing": [],
    }

    for i, line in enumerate(grid):
        for j, cell in enumerate(line):
            if cell != "#":
                coords["cells"].append((i, j))
            if cell in [" ", "G", "T", "W", "C", "R", "N"]:
                coords["empty"].append((i, j))
            if cell == "H":
                coords["hero"].append((i, j))
            elif cell == "G":
                coords["guards"].append((i, j))
            elif cell == "T":
                coords["targets"].append((i, j))
            elif cell == "W":
                coords["walls"].append((i, j))
            elif cell == "C":
                coords["costumes"].append((i, j))
            elif cell == "R":
                coords["ropes"].append((i, j))
            elif cell == "N":
                coords["nothing"].append((i, j))

    return coords


def vocabulary(coords: dict, t_max: int) -> dict:
    """
    :param coords: dict containing coord of each element of the map
    :param t_max: horizon
    :return: dict containing all the vocabulary
    """
    cells = coords["cells"]
    targets = coords["targets"]

    act_vars = [("do", t, a) for t in range(t_max) for a in ACTIONS]
    at_vars = [("at", t, c) for t in range(t_max + 1) for c in cells]
    vision_vars = [("vision", t, c) for t in range(t_max + 1) for c in cells]
    hear_vars = [("hear", t, c) for t in range(t_max + 1) for c in cells]
    hero_vars = [("hero", t, c) for t in range(t_max + 1) for c in cells]
    guard_vars = [("guard", t, c) for t in range(t_max + 1) for c in cells]
    target_vars = [("target", t, c) for t in range(t_max + 1) for c in targets]
    wall_vars = [("wall", t, c) for t in range(t_max + 1) for c in cells]
    costume_vars = [("costume", t, c) for t in range(t_max + 1) for c in cells]
    rope_vars = [("rope", t, c) for t in range(t_max + 1) for c in cells]
    nothing_vars = [("nothing", t, c) for t in range(t_max + 1) for c in cells]

    return {
        v: i + 1
        for i, v in enumerate(
            act_vars
            + at_vars
            + vision_vars
            + hear_vars
            + hero_vars
            + guard_vars
            + target_vars
            + wall_vars
            + costume_vars
            + rope_vars
            + nothing_vars
        )
    }




#cell_to_vaiable
def cell_to_variable(i: int, j: int, val: int) -> PropositionnalVariable:
    return i * 42 + j * 7 + val + 1


#variable_to_cell
def variable_to_cell(var: PropositionnalVariable) -> Tuple[int, int, int]:
    val = ((var - 1) // 42) % 13 + 1
    return ((var-1)//42 ,((var-1) % 42)//7, val)

#variable_to_clause

def at_least_one(variables: List[PropositionnalVariable]) -> Clause:
    clauses: Clause = []
    for i in variables:
        clauses.append(i)
    return clauses

def unique(variables: List[PropositionnalVariable]) -> ClauseBase:
    clauseBase: ClauseBase = []
    clauseBase.append(at_least_one(variables))

    for i, j in combinations(variables, 2):
        clauseBase.append([-i, -j])

    return clauseBase

#
def create_cell_constraints() -> Clause:
    clauseBase: Clause = []
    for i in range(6):
        for j in range(7):
            variables: List[PropositionnalVariable] = []
            for val in range(1,14):
                variables.append(cell_to_variable(i, j, val))

            clauseBase += unique(variables)

    return clauseBase


def create_line_constraints() -> Clause:
    clauseBase: Clause = []

    for i in range(6):
        for val in range(7):
            variables: List[PropositionnalVariable] = []
            for j in range(1,14):
                variables.append(cell_to_variable(i, j, val))

            clauseBase.append(at_least_one(variables))

    return clauseBase

def create_line_constraints() ->ClauseBase:
    clauseBase: ClauseBase = []

    for i in range(9):
        for val in range(9):
            variables: List[PropositionnalVariable] = []
            for j in range(9):
                variables.append(cell_to_variable(i, j, val))

            clauseBase.append(at_least_one(variables))

    return clauseBase


def create_column_constraints() -> ClauseBase:
    clauseBase: ClauseBase = []

    for j in range(9):
        for val in range(9):
            variables: List[PropositionnalVariable] = []
            for i in range(9):
                variables.append(cell_to_variable(i, j, val))

            clauseBase.append(at_least_one(variables))

    return clauseBase

def create_vision_constraints()-> ClauseBase:
    clauseBase: ClauseBase =[]



