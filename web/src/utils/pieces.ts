import type { Coordinate, PieceDescriptor, PlayerColor } from "../types/blokus";

type PieceCell = [number, number];

function normalizeCells(cells: PieceCell[]): PieceCell[] {
  const minRow = Math.min(...cells.map(([row]) => row));
  const minCol = Math.min(...cells.map(([, col]) => col));
  return [...cells]
    .map(([row, col]) => [row - minRow, col - minCol] as PieceCell)
    .sort(([leftRow, leftCol], [rightRow, rightCol]) =>
      leftRow === rightRow ? leftCol - rightCol : leftRow - rightRow
    );
}

function rotateCells(cells: PieceCell[]): PieceCell[] {
  return cells.map(([row, col]) => [col, -row]);
}

function reflectCells(cells: PieceCell[]): PieceCell[] {
  return cells.map(([row, col]) => [row, -col]);
}

export function transformPieceCells(
  cells: PieceCell[],
  rotation: number,
  reflection: boolean
): PieceCell[] {
  let transformed = reflection ? reflectCells(cells) : [...cells];
  const turns = ((rotation % 360) + 360) % 360 / 90;
  for (let turn = 0; turn < turns; turn += 1) {
    transformed = rotateCells(transformed);
  }
  return normalizeCells(transformed);
}

export function pieceCellsForId(
  pieces: PieceDescriptor[],
  pieceId: string,
  rotation = 0,
  reflection = false
): PieceCell[] {
  const piece = pieces.find((candidate) => candidate.piece_id === pieceId);
  if (!piece) {
    return [];
  }
  return transformPieceCells(piece.cells, rotation, reflection);
}

export function placePieceAtAnchor(anchor: Coordinate, cells: PieceCell[]): Coordinate[] {
  return cells.map(([row, col]) => ({
    row: anchor.row + row,
    col: anchor.col + col
  }));
}

export function coordinatesToKey(row: number, col: number): string {
  return `${row}:${col}`;
}

export function coordinatesToSet(cells: Coordinate[]): Set<string> {
  return new Set(cells.map((cell) => coordinatesToKey(cell.row, cell.col)));
}

export function isPlacementValid(
  board: (PlayerColor | null)[][],
  placement: Coordinate[]
): boolean {
  return placement.every(({ row, col }) => {
    if (row < 0 || row >= board.length || col < 0 || col >= board[row].length) {
      return false;
    }
    return board[row][col] === null;
  });
}
