"""Hand-written tokenizer and recursive-descent parser for factor expressions.

Grammar (lowest to highest precedence)::

    expr    := term (("+" | "-") term)*
    term    := unary (("*" | "/") unary)*
    unary   := "-" unary | primary
    primary := NUMBER | IDENT | IDENT "(" [expr ("," expr)*] ")" | "(" expr ")"

Infix ``+ - * /`` desugar to ``add/sub/mul/div`` and unary minus to ``neg``
(or to a negative literal when applied to a number).  Identifiers are
case-insensitive and normalized to lower case.  The parser is purely
syntactic: operator names, arities and argument kinds are checked by
:mod:`llm_factor_mining.dsl.validate`.  User text is never passed to
``eval``/``exec`` or any Python-literal parser.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal

from .nodes import Call, Constant, Node, Terminal


MAX_EXPRESSION_LENGTH = 2000
MAX_NESTING = 48

TokenKind = Literal[
    "NUMBER", "IDENT", "LPAREN", "RPAREN", "COMMA", "PLUS", "MINUS", "STAR", "SLASH", "EOF"
]

_NUMBER = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PUNCTUATION: dict[str, TokenKind] = {
    "(": "LPAREN",
    ")": "RPAREN",
    ",": "COMMA",
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
}
_BINARY_OPS: dict[TokenKind, str] = {"PLUS": "add", "MINUS": "sub", "STAR": "mul", "SLASH": "div"}


class ParseError(ValueError):
    """Syntax error with the 0-based character ``position`` of the offending token."""

    def __init__(self, message: str, position: int, text: str) -> None:
        self.message = message
        self.position = int(position)
        self.text = text
        super().__init__(f"{message} at position {self.position}")

    def pretty(self) -> str:
        """Return the message with a caret under the offending character."""

        excerpt = self.text if len(self.text) <= 160 else self.text[:157] + "..."
        caret = " " * min(self.position, len(excerpt)) + "^"
        return f"{self}\n  {excerpt}\n  {caret}"


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    text: str
    position: int


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into tokens, ending with an ``EOF`` token."""

    tokens: list[Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in _PUNCTUATION:
            tokens.append(Token(_PUNCTUATION[char], char, index))
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < length and text[index + 1].isdigit()):
            match = _NUMBER.match(text, index)
            if match is None:  # pragma: no cover - guarded by the condition above
                raise ParseError("malformed number", index, text)
            end = match.end()
            if end < length and (text[end].isalnum() or text[end] in "._"):
                raise ParseError("malformed number", index, text)
            tokens.append(Token("NUMBER", match.group(0), index))
            index = end
            continue
        if char.isalpha() or char == "_":
            match = _IDENT.match(text, index)
            if match is None:  # pragma: no cover - guarded by the condition above
                raise ParseError("malformed identifier", index, text)
            tokens.append(Token("IDENT", match.group(0), index))
            index = match.end()
            continue
        raise ParseError(f"unexpected character {char!r}", index, text)
    tokens.append(Token("EOF", "", length))
    return tokens


class _Parser:
    def __init__(self, text: str, tokens: list[Token], max_nesting: int) -> None:
        self.text = text
        self.tokens = tokens
        self.index = 0
        self.depth = 0
        self.max_nesting = max_nesting

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.tokens[self.index]
        if token.kind != "EOF":
            self.index += 1
        return token

    def expect(self, kind: TokenKind, description: str) -> Token:
        token = self.current
        if token.kind != kind:
            found = "end of input" if token.kind == "EOF" else repr(token.text)
            raise ParseError(f"expected {description}, found {found}", token.position, self.text)
        return self.advance()

    def enter(self, position: int) -> None:
        self.depth += 1
        if self.depth > self.max_nesting:
            raise ParseError(
                f"expression nesting exceeds {self.max_nesting} levels", position, self.text
            )

    def leave(self) -> None:
        self.depth -= 1

    def parse(self) -> Node:
        if self.current.kind == "EOF":
            raise ParseError("empty expression", 0, self.text)
        node = self.expr()
        token = self.current
        if token.kind != "EOF":
            raise ParseError(f"unexpected token {token.text!r}", token.position, self.text)
        return node

    def expr(self) -> Node:
        self.enter(self.current.position)
        node = self.term()
        while self.current.kind in ("PLUS", "MINUS"):
            op = _BINARY_OPS[self.advance().kind]
            node = Call(op, (node, self.term()))
        self.leave()
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.current.kind in ("STAR", "SLASH"):
            op = _BINARY_OPS[self.advance().kind]
            node = Call(op, (node, self.unary()))
        return node

    def unary(self) -> Node:
        if self.current.kind == "MINUS":
            token = self.advance()
            self.enter(token.position)
            operand = self.unary()
            self.leave()
            if isinstance(operand, Constant):
                return Constant(-operand.value, operand.integer)
            return Call("neg", (operand,))
        return self.primary()

    def primary(self) -> Node:
        token = self.current
        if token.kind == "NUMBER":
            self.advance()
            return _number(token, self.text)
        if token.kind == "IDENT":
            self.advance()
            name = token.text.lower()
            if self.current.kind != "LPAREN":
                return Terminal(name)
            self.advance()
            args: list[Node] = []
            if self.current.kind != "RPAREN":
                args.append(self.expr())
                while self.current.kind == "COMMA":
                    self.advance()
                    args.append(self.expr())
            self.expect("RPAREN", f"',' or ')' to close {name}(")
            return Call(name, tuple(args))
        if token.kind == "LPAREN":
            self.advance()
            node = self.expr()
            self.expect("RPAREN", "')'")
            return node
        found = "end of input" if token.kind == "EOF" else repr(token.text)
        raise ParseError(f"expected a number, name or '(', found {found}", token.position, self.text)


def _number(token: Token, text: str) -> Constant:
    literal = token.text
    integer = not any(marker in literal for marker in ".eE")
    value = float(literal)
    if not math.isfinite(value):
        raise ParseError("numeric literal is not finite", token.position, text)
    return Constant(value, integer)


def parse(
    text: str,
    *,
    max_length: int = MAX_EXPRESSION_LENGTH,
    max_nesting: int = MAX_NESTING,
) -> Node:
    """Parse ``text`` into an immutable AST or raise :class:`ParseError`."""

    if not isinstance(text, str):
        raise TypeError("expression must be a string")
    if len(text) > max_length:
        raise ParseError(f"expression longer than {max_length} characters", max_length, text)
    return _Parser(text, tokenize(text), max_nesting).parse()
