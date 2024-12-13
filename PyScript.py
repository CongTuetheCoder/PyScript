from enum import Enum
from functools import partial
from typing import Self, IO
from random import random, randint
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QDialog, QLineEdit, QPushButton, QLabel
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextDocument
from PyQt6.QtCore import QRegularExpression, Qt, QEvent, QEventLoop, QObject
from PyQt6 import uic
import os
import sys
import json

## SETUP
sys.set_int_max_str_digits(10000)
input_pos: int = -1
with open("accounts.json", "r") as aFile:
	accounts: dict = json.loads("".join(aFile))

## TOKEN TYPES
class TT(Enum):
	EOF = 0; NEWLINE = 1
	INTEGER = 2; REAL = 3
	OPERATION = 4
	LPAREN = 5; RPAREN = 6; LBRACE = 7; RBRACE = 8; LSQUARE = 9; RSQUARE = 10
	IDENTIFIER = 11; KEYWORD = 12
	COLON = 13; COMMA = 14
	LARROW = 15; RARROW = 16
	STRING = 17; CHAR = 18
	QUESTION = 19; DOT = 20

	def __str__(self):
		return super().__str__().replace("TT.", "")

## TOKEN
class Token:
	def __init__(self, tokenType: TT, value = None):
		self.type = tokenType
		self.value = value
	
	def __repr__(self) -> str:
		return f"{self.type}:{self.value}" if self.value else f"{self.type}"
	
	def __eq__(self, value) -> bool:
		if isinstance(value, Token):
			return self.type == value.type and self.value == value.value
		return False
	
	def __ne__(self, value):
		return not (self == value)

## ERROR
class Error:
	def __init__(self, errorName: str, details: str):
		self.errorName = errorName
		self.details = details
	
	def __repr__(self):
		return f"{self.errorName}: {self.details}\n"

class IllegalCharError(Error):
	def __init__(self, details):
		super().__init__("Illegal Character", details)

class ExpectedCharError(Error):
	def __init__(self, details):
		super().__init__("Expected Character", details)

class InvalidSyntaxError(Error):
	def __init__(self, details):
		super().__init__("Invalid Syntax", details)

class RTError(Error):
	def __init__(self, details):
		super().__init__("Runtime Error", details)

## LEXER
class Lexer:
	def __init__(self, fn: str, ftxt: str):
		self.fn: str = fn
		self.ftxt: str = ftxt
		self.pos: int = -1
		self.currentChar: str|None = None

		self.KEYWORDS: list[str] = [
			"declare", "const", # Variable declaration
			"integer", "real", "boolean", "string", "char", "array", "file", # Data types
			"for", "step", "while", "repeat", "until", # Loops
			"if", "elif", "else", # Conditionals
			"object", "method", # Objects
			"function", "inline", # Functions
			"try", "catch", "otherwise", # Try-catch statements
			"switch", "case", # Switch statements
			"and", "or", "not", # Logical operators
			"return", "break", "continue" # Exit keywords
		]

		self.advance()
	
	def advance(self):
		self.pos += 1
		self.currentChar = self.ftxt[self.pos] if self.pos < len(self.ftxt) else None
	
	def lex(self):
		tokens: list[Token] = []
		escapes: dict[str, str] = {
			"t": "\t", "n": "\n", "b": "\b"
		}
		while self.currentChar != None:
			match self.currentChar:
				case " ": self.advance()
				case "\t": self.advance()
				case "\n": tokens.append(Token(TT.NEWLINE)); self.advance()
				case "\r": tokens.append(Token(TT.NEWLINE)); self.advance()
				case ";": tokens.append(Token(TT.NEWLINE)); self.advance()
				case "(": tokens.append(Token(TT.LPAREN)); self.advance()
				case ")": tokens.append(Token(TT.RPAREN)); self.advance()
				case "[": tokens.append(Token(TT.LSQUARE)); self.advance()
				case "]": tokens.append(Token(TT.RSQUARE)); self.advance()
				case "{": tokens.append(Token(TT.LBRACE)); self.advance()
				case "}": tokens.append(Token(TT.RBRACE)); self.advance()
				case ":": tokens.append(Token(TT.COLON)); self.advance()
				case ",": tokens.append(Token(TT.COMMA)); self.advance()
				case "?": tokens.append(Token(TT.QUESTION)); self.advance()
				case ".": tokens.append(Token(TT.DOT)); self.advance()
				case "'":
					self.advance()
					if self.currentChar == "'":
						self.advance()
						tokens.append(Token(TT.CHAR, '\0'))
					elif self.currentChar == "\\":
						self.advance()
						escape_char = escapes.get(self.currentChar, self.currentChar)
						self.advance()
						if self.currentChar != "'":
							return None, ExpectedCharError("Expected terminating apostrophe.")
						self.advance()
						tokens.append(Token(TT.CHAR, escape_char))
					else:
						char = self.currentChar
						self.advance()
						if self.currentChar != "'":
							return None, ExpectedCharError("Expected terminating apostrophe.")
						self.advance()
						tokens.append(Token(TT.CHAR, char))
				case "<":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "<="))
					elif self.currentChar == "-":
						self.advance()
						tokens.append(Token(TT.LARROW))
					elif self.currentChar == "<":
						self.advance()
						tokens.append(Token(TT.OPERATION, "<<"))
					else:
						tokens.append(Token(TT.OPERATION, "<"))
				case ">":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, ">="))
					elif self.currentChar == ">":
						self.advance()
						tokens.append(Token(TT.OPERATION, ">>"))
					else:
						tokens.append(Token(TT.OPERATION, ">"))
				case "=":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "=="))
					else:
						tokens.append(Token(TT.OPERATION, "="))
				case "+":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "+="))
					else:
						tokens.append(Token(TT.OPERATION, "+"))
				case "*":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "*="))
					else:
						tokens.append(Token(TT.OPERATION, "*"))
				case "%":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "%="))
					else:
						tokens.append(Token(TT.OPERATION, "%"))
				case "^":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "^="))
					else:
						tokens.append(Token(TT.OPERATION, "^"))
				case "!":
					self.advance()
					if self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "!="))
					else:
						return None, ExpectedCharError("Expected '=' (after '!').")
				case "-":
					self.advance()
					if self.currentChar == ">":
						self.advance()
						tokens.append(Token(TT.RARROW))
					elif self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "-="))
					else:
						tokens.append(Token(TT.OPERATION, "-"))
				case "/":
					self.advance()
					if self.currentChar == "/": # Skip comments
						while self.currentChar not in [None, "\n", "\r"]:
							self.advance()
					elif self.currentChar == "=":
						self.advance()
						tokens.append(Token(TT.OPERATION, "/="))
					else:
						tokens.append(Token(TT.OPERATION, "/"))
				case "\"":
					# Make string
					self.advance()
					string: str = ""
					escape: bool = False
					while self.currentChar != None and self.currentChar not in "\n\r\"" or (self.currentChar == "\"" and escape):
						if escape:
							string += escapes.get(self.currentChar, self.currentChar)
							escape = False
						else:
							if self.currentChar == "\\":
								escape = True
							else:
								string += self.currentChar
								escape = False
						self.advance()
					if self.currentChar != "\"":
						return None, ExpectedCharError("Expected '\"' at the end of the string.")
					self.advance()
					tokens.append(Token(TT.STRING, string))
				case _:
					if self.currentChar.isdigit(): # Number
						numStr: str = ""
						dotCount: int = 0
						while self.currentChar != None and (self.currentChar.isdigit() or self.currentChar == "."):
							if self.currentChar == ".":
								dotCount += 1
								if dotCount > 1:
									break
							numStr += self.currentChar
							self.advance()
						tokens.append(Token(TT.INTEGER, int(numStr)) if dotCount == 0 else Token(TT.REAL, float(numStr)))
					elif self.currentChar.isalpha() or self.currentChar == "_": # Identifier / keyword
						idStr = ""
						while self.currentChar != None and (self.currentChar.isalpha() or self.currentChar.isdigit() or self.currentChar == "_"):
							idStr += self.currentChar
							self.advance()
						tokens.append(Token(TT.KEYWORD if idStr in self.KEYWORDS else TT.IDENTIFIER, idStr))
					else:
						return None, IllegalCharError(f"'{self.currentChar}'")
		tokens.append(Token(TT.EOF))
		return tokens, None

## AST NODES
class Expression: ...

class Program:
	def __init__(self, body: list[Expression]):
		self.body = body

	def __repr__(self):
		result = "Program {\n"
		for line in self.body:
			result += f"\t{line}\n"
		return result + "}"

class TernaryOperation(Expression):
	def __init__(self, condition: Expression, truthyValue: Expression, falsyValue: Expression):
		self.condition = condition
		self.truthyValue = truthyValue
		self.falsyValue = falsyValue
	
	def __repr__(self):
		return f"TernaryOp({self.condition} ? {self.truthyValue} : {self.falsyValue})"

class BinaryOperation(Expression):
	def __init__(self, left: Expression, operator: Token, right: Expression):
		self.left = left
		self.operator = operator
		self.right = right
	
	def __repr__(self):
		return f"BinaryOp({self.left}, {self.operator}, {self.right})"

class UnaryOperation(Expression):
	def __init__(self, operator: Token, expr: Expression):
		self.operator = operator
		self.expr = expr
	
	def __repr__(self):
		return f"UnaryOp({self.operator}, {self.expr})"

class IntLiteral(Expression):
	def __init__(self, value: int):
		self.value = value
	
	def __repr__(self):
		return f"INT({self.value})"

class RealLiteral(Expression):
	def __init__(self, value: float):
		self.value = value
	
	def __repr__(self):
		return f"REAL({self.value})"

class StringLiteral(Expression):
	def __init__(self, value: str):
		self.value = value
	
	def __repr__(self):
		return f'STRING("{self.value}")'

	def __str__(self):
		return f'STRING("{self.value}")'

class CharLiteral(Expression):
	def __init__(self, value: str):
		self.value = value
	
	def __repr__(self):
		return f"CHAR('{self.value}')"

class Property(Expression):
	def __init__(self, key: str, dataType: str, value: Expression):
		self.key = key
		self.dataType = dataType
		self.value = value

class ObjectLiteral(Expression):
	def __init__(self, symbol: str, properties: list[Property]):
		self.symbol = symbol
		self.properties = properties

class ArrayLiteral(Expression):
	def __init__(self, values: list[Expression]):
		self.values = values
	
	def __repr__(self):
		return f"ARRAY{self.values}"

class Identifier(Expression):
	def __init__(self, value: str):
		self.value = value
	
	def __repr__(self):
		return f"IDENTIFIER({self.value})"

class VarDeclaration(Expression):
	def __init__(self, symbol: str, isConst: bool, dataType: str, value: Expression|None = None):
		self.symbol = symbol
		self.isConst = isConst
		self.dataType = dataType
		self.value = value
	
	def __repr__(self):
		if self.isConst: return f"Declare(Const {self.symbol}: {self.dataType} -> {self.value})" if self.value else f"Declare(Const {self.symbol}: {self.dataType})"
		return f"Declare(Var {self.symbol}: {self.dataType} -> {self.value})" if self.value else f"Declare(Var {self.symbol}: {self.dataType})"

class Assignment(Expression):
	def __init__(self, symbol: str, value: Expression, operation: str):
		self.symbol = symbol
		self.value = value
		self.operation = operation
	
	def __repr__(self):
		return f"Assign({self.symbol} {self.operation} {self.value})"

class ForExpression(Expression):
	def __init__(self, symbol: str, body: list[Expression], startValue: Expression, endValue: Expression, stepValue: Expression|None = None):
		self.symbol = symbol
		self.body = body
		self.startValue = startValue
		self.endValue = endValue
		self.stepValue = stepValue

class WhileExpression(Expression):
	def __init__(self, condition: Expression, body: list[Expression]):
		self.condition = condition
		self.body = body

class RepeatExpression(Expression):
	def __init__(self, condition: Expression, body: list[Expression]):
		self.condition = condition
		self.body = body

class IfExpression(Expression):
	def __init__(self, cases: list[tuple[Expression, list[Expression]]], elseCase: list[Expression]|None = None):
		self.cases = cases
		self.elseCase = elseCase

class FunctionDeclaration(Expression):
	def __init__(self, functionName: str, args: list[tuple[str, str]], body: list[Expression], should_auto_return: bool):
		self.functionName = functionName
		self.args = args
		self.body = body
		self.should_auto_return = should_auto_return
	
	def __repr__(self):
		result: str = f"FunctionDeclaration {self.functionName} ({', '.join([arg + ':' + data_type for arg, data_type in self.args])}) " + "{\n"
		for line in self.body:
			result += f"\t\t{line}\n"
		return result + "\t}"

class MethodDeclaration(Expression):
	def __init__(self, methodName: str, args: list[tuple[str, str]], body: list[Expression], obj: ObjectLiteral):
		self.methodName = methodName
		self.args = args
		self.body = body
		self.obj = obj

class MemberExpression(Expression):
	def __init__(self, object: Expression, property: Identifier):
		self.object = object
		self.property = property

class CallExpression(Expression):
	def __init__(self, identifier: Identifier, args: list[Expression]):
		self.identifier = identifier
		self.args = args
	
	def __str__(self):
		return f"CALL {self.identifier.value}{self.args}"

class TryExpression(Expression):
	def __init__(self, tryBody: list[Expression], catchBody: list[Expression], otherwiseBody: list[Expression]):
		self.tryBody = tryBody
		self.catchBody = catchBody
		self.otherwiseBody = otherwiseBody

class SwitchStatement(Expression):
	def __init__(self, parameter: Expression, cases: list[tuple[Expression, Program]], otherwise_case: Program|None):
		self.parameter = parameter
		self.cases = cases
		self.otherwise_case = otherwise_case

class ArrayAccessExpression(Expression):
	def __init__(self, array: Identifier|ArrayLiteral, index: Expression, end_index: Expression|None = None):
		self.array = array
		self.index = index
		self.end_index = end_index
	
	def __repr__(self):
		return f"{self.array}[{self.index}" + f":{self.end_index}]" if self.end_index else "]"

class ArraySetExpression(Expression):
	def __init__(self, array: Identifier|ArrayLiteral, index: Expression, operation: str, value: Expression):
		self.array = array
		self.index = index
		self.operation = operation
		self.value = value
	
	def __repr__(self):
		return f"{self.array}[{self.index}] {self.operation} {self.value}"

class ReturnNode(Expression):
	def __init__(self, node_to_return: Expression|None):
		self.node_to_return = node_to_return
	
	def __repr__(self):
		return f"Return ({self.node_to_return})"

class ContinueNode(Expression):
	def __repr__(self):
		return f"continue"

class BreakNode(Expression):
	def __repr__(self):
		return f"break"

## PARSE RESULT
class ParseResult:
	def __init__(self):
		self.node: Expression|None = None
		self.error: Error|None = None
		self.advance_count: int = 0
		self.to_reverse_count: int = 0

	def register(self, res: Self):
		if isinstance(res, ParseResult):
			if res.error: self.error = res.error
			self.advance_count += res.advance_count
		return res.node
	
	def try_register(self, res: Self):
		if res.error:
			self.to_reverse_count = res.advance_count
			return None
		return self.register(res)
	
	def success(self, node: Program):
		self.node = node
		return self
	
	def fail(self, error: Error):
		self.error = error
		return self

## PARSER
class Parser:
	def __init__(self, fn: str, tokens: list[Token]):
		self.fn = fn
		self.tokens = tokens
		self.tokenIndex: int = -1
		self.currentToken: Token|None = None
		self.dataTypes = ["integer", "real", "boolean", "string", "char", "object", "file", "function", "array"]
		self.advance(None)
	
	def advance(self, pr: ParseResult|None):
		self.tokenIndex += 1
		if isinstance(pr, ParseResult): pr.advance_count += 1
		if self.tokenIndex < len(self.tokens): self.currentToken = self.tokens[self.tokenIndex]
		return self.currentToken
	
	def step_back(self, amount: int = 1):
		self.tokenIndex -= amount
		if self.tokenIndex >= 0: self.currentToken = self.tokens[self.tokenIndex]
		return self.currentToken
	
	def parse(self):
		pr = ParseResult()
		ast = Program([])
		while self.currentToken.type != TT.EOF:
			line = pr.register(self.statement())
			if pr.error: return pr
			ast.body.append(line)
			# Skip newlines
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		return pr.success(ast)
	
	def statement(self):
		pr = ParseResult()
		while self.currentToken.type == TT.NEWLINE: self.advance(pr) # Advance past newlines
		if self.currentToken.type == TT.KEYWORD:
			if self.currentToken.value == "declare": return self.var_declaration()
			elif self.currentToken.value == "const": return self.var_declaration()
			elif self.currentToken.value == "for": return self.for_loop()
			elif self.currentToken.value == "while": return self.while_loop()
			elif self.currentToken.value == "repeat": return self.repeat_loop()
			elif self.currentToken.value == "if": return self.if_statement()
			elif self.currentToken.value == "object": return self.object_declaration()
			elif self.currentToken.value == "function": return self.function_declaration()
			elif self.currentToken.value == "try": return self.try_catch_statement()
			elif self.currentToken.value == "switch": return self.switch_statement()
			elif self.currentToken.value == "return": return self.return_statement()
			elif self.currentToken.value == "continue": return self.continue_statement()
			elif self.currentToken.value == "break": return self.break_statement()
			else: return self.expression()
		else: return self.expression()

	def return_statement(self):
		pr = ParseResult()
		self.advance(pr)

		expr = pr.try_register(self.expression())
		if not expr: self.step_back(pr.to_reverse_count)
		return pr.success(ReturnNode(expr))
	
	def continue_statement(self):
		pr = ParseResult()
		self.advance(pr)
		return pr.success(ContinueNode())

	def break_statement(self):
		pr = ParseResult()
		self.advance(pr)
		return pr.success(BreakNode())

	def switch_statement(self):
		pr = ParseResult()
		self.advance(pr)
		parameter = pr.register(self.expression())
		cases: list[tuple[Expression, Program]] = []
		otherwise_case: Program|None = None
		if pr.error: return pr
		
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after parameter."))
		self.advance(pr)

		while self.currentToken.type == TT.NEWLINE: self.advance(pr)

		# Register first case
		if self.currentToken not in (Token(TT.KEYWORD, "case"), Token(TT.KEYWORD, "otherwise")):
			return pr.fail(InvalidSyntaxError("Expected 'case' or 'otherwise' keyword after switch initialization."))
		
		# Register cases
		test_values: int = 0
		while self.currentToken.type not in (TT.EOF, TT.RBRACE):
			if self.currentToken == Token(TT.KEYWORD, "case"):
				
				# Check if there is already an otherwise case
				if otherwise_case != None:
					return pr.fail(InvalidSyntaxError("Unexpected case statement after otherwise statement."))

				self.advance(pr)
				test_data = pr.register(self.expression())
				if pr.error: return pr
				if self.currentToken.type != TT.LBRACE:
					return pr.fail(InvalidSyntaxError("Expected '{' before test case #" + str(test_values) + "."))
				self.advance(pr)

				body: list[Expression] = []
				while self.currentToken.type not in (TT.EOF, TT.RBRACE):
					stmt = pr.register(self.statement())
					if pr.error: return pr
					body.append(stmt)
					# Skip newlines
					while self.currentToken.type == TT.NEWLINE: self.advance(pr)	

				if self.currentToken.type != TT.RBRACE:
					return pr.fail(InvalidSyntaxError("Expected '}' after test case #" + str(test_values) + "."))
				self.advance(pr)
				test_values += 1
				cases.append((test_data, Program(body)))

			elif self.currentToken == Token(TT.KEYWORD, "otherwise"):
				self.advance(pr)
				if self.currentToken.type != TT.LBRACE:
					return pr.fail(InvalidSyntaxError("Expected '{' before otherwise case."))
				self.advance(pr)

				body: list[Expression] = []
				while self.currentToken.type not in (TT.EOF, TT.RBRACE):
					stmt = pr.register(self.statement())
					if pr.error: return pr
					body.append(stmt)
					# Skip newlines
					while self.currentToken.type == TT.NEWLINE: self.advance(pr)	

				if self.currentToken.type != TT.RBRACE:
					return pr.fail(InvalidSyntaxError("Expected '}' after otherwise case."))
				self.advance(pr)
				otherwise_case = Program(body)
			
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		
		# End switch statement
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after switch statement."))
		self.advance(pr)
		return pr.success(SwitchStatement(parameter, cases, otherwise_case))

	def try_catch_statement(self):
		pr = ParseResult()
		self.advance(pr) # Advances past the 'try' keyword
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after 'try' KEYWORD."))
		self.advance(pr)
		tryBody: list[Expression] = []
		while self.currentToken.type not in (TT.EOF, TT.RBRACE):
			stmt = pr.register(self.statement())
			if pr.error: return pr
			tryBody.append(stmt)
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after try statement body."))
		self.advance(pr)

		# Catch
		while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken != Token(TT.KEYWORD, "catch"):
			return pr.fail(InvalidSyntaxError("Expected 'catch' after try statement body."))
		self.advance(pr)
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after 'catch' KEYWORD."))
		self.advance(pr)
		catchBody: list[Expression] = []
		while self.currentToken.type not in (TT.EOF, TT.RBRACE):
			stmt = pr.register(self.statement())
			if pr.error: return pr
			catchBody.append(stmt)
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after catch statement body."))
		self.advance(pr)

		# Otherwise
		while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		otherwiseBody: list[Expression] = []
		if self.currentToken == Token(TT.KEYWORD, "otherwise"):
			self.advance(pr)
			if self.currentToken.type != TT.LBRACE:
				return pr.fail(InvalidSyntaxError("Expected '{' after 'otherwise' KEYWORD."))
			self.advance(pr)
			while self.currentToken.type not in (TT.EOF, TT.RBRACE):
				stmt = pr.register(self.statement())
				if pr.error: return pr
				otherwiseBody.append(stmt)
				while self.currentToken.type == TT.NEWLINE: self.advance(pr)
			if self.currentToken.type != TT.RBRACE:
				return pr.fail(InvalidSyntaxError("Expected '}' after otherwise statement body."))
			self.advance(pr)
		return pr.success(TryExpression(tryBody, catchBody, otherwiseBody))

	def check_for_data_type(self):
		pr = ParseResult()
		if not(self.currentToken.type == TT.KEYWORD and self.currentToken.value in self.dataTypes):
			return pr.fail(InvalidSyntaxError(f"Expected a data type."))
		data_type = self.currentToken.value
		self.advance(pr)

		if data_type == "array":
			if self.currentToken.type != TT.LSQUARE:
				return pr.fail(InvalidSyntaxError("Expected '[' after 'array' KEYWORD."))
			self.advance(pr)
			if not (self.currentToken.type == TT.KEYWORD and self.currentToken.value in self.dataTypes[:-1]):
				return pr.fail(InvalidSyntaxError(f"Expected a data type."))
			data_type += f"[{self.currentToken.value}]"
			self.advance(pr)
			if self.currentToken.type != TT.RSQUARE:
				return pr.fail(InvalidSyntaxError("Expected ']' after array type."))
			self.advance(pr)
		return pr.success(data_type)

	def var_declaration(self):
		pr = ParseResult()
		isConst: bool = (self.currentToken.value == "const")
		self.advance(pr)

		# Variable / constant name
		if self.currentToken.type != TT.IDENTIFIER:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFER after 'declare'/'const' KEYWORD."))
		symbol = self.currentToken.value
		self.advance(pr)
		if self.currentToken.type != TT.COLON:
			return pr.fail(InvalidSyntaxError("Expected ':' after symbol."))
		self.advance(pr)

		# Check for data types
		data_type = self.check_for_data_type()
		if data_type.error: return data_type
		
		if self.currentToken.type in (TT.NEWLINE, TT.EOF):
			# End of variable declaration
			if isConst:
				return pr.fail(InvalidSyntaxError(f"Constant '{symbol}' must be assigned to a value."))
			return pr.success(VarDeclaration(symbol, False, data_type.node))
		if self.currentToken != Token(TT.OPERATION, "="):
			return pr.fail(InvalidSyntaxError("Expected '=' after variable/constant name."))
		self.advance(pr)
		value = pr.register(self.expression())
		if pr.error: return pr
		return pr.success(VarDeclaration(symbol, isConst, data_type.node, value))
	
	def function_declaration(self):
		pr = ParseResult()
		self.advance(pr)
		if self.currentToken.type != TT.IDENTIFIER:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER (as a function name)."))
		functionName: str = self.currentToken.value
		self.advance(pr)
		if self.currentToken.type != TT.LPAREN:
			return pr.fail(InvalidSyntaxError("Expected '(' after function name."))
		self.advance(pr)
		args: list[tuple[str, str]] = []
		if self.currentToken.type == TT.IDENTIFIER:
			arg = self.currentToken.value
			self.advance(pr)
			if self.currentToken.type != TT.COLON:
				return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
			self.advance(pr)
			if self.currentToken.type != TT.KEYWORD or (self.currentToken.value not in self.dataTypes):
				return pr.fail(InvalidSyntaxError(f"Expected a data type."))
			dataType = self.currentToken.value
			self.advance(pr)
			args.append((arg, dataType))
			while self.currentToken.type == TT.COMMA:
				self.advance(pr)
				if self.currentToken.type != TT.IDENTIFIER:
					return pr.fail(InvalidSyntaxError("Expected IDENTIFIER (as a parameter)."))
				arg = self.currentToken.value
				self.advance(pr)
				if self.currentToken.type != TT.COLON:
					return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
				self.advance(pr)
				if self.currentToken.type != TT.KEYWORD or (self.currentToken.value not in self.dataTypes):
					return pr.fail(InvalidSyntaxError(f"Expected a data type."))
				dataType = self.currentToken.value
				self.advance(pr)
				args.append((arg, dataType))
		
		# Check for right parenthesis
		if self.currentToken.type != TT.RPAREN:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER or ')'."))
		self.advance(pr)

		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after function declaration."))
		self.advance(pr)

		body: list[Expression] = []
		while self.currentToken.type == TT.NEWLINE: self.advance(pr)

		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after function body."))
		self.advance(pr)
		return pr.success(FunctionDeclaration(functionName, args, body, False))

	def for_loop(self):
		pr = ParseResult()
		self.advance(pr) # Advances past the 'for' keyword
		if self.currentToken.type != TT.IDENTIFIER:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER after 'for' keyword."))
		symbol = self.currentToken.value
		self.advance(pr)
		if self.currentToken != Token(TT.OPERATION, "="):
			return pr.fail(InvalidSyntaxError("Expected '=' after iterator."))
		self.advance(pr)
		startValue = pr.register(self.expression())
		if pr.error: return pr
		if self.currentToken.type != TT.RARROW:
			return pr.fail(InvalidSyntaxError("Expected '->' after start value."))
		self.advance(pr)
		endValue = pr.register(self.expression())
		if pr.error: return pr
		if self.currentToken == Token(TT.KEYWORD, "step"):
			self.advance(pr)
			stepValue = pr.register(self.expression())
			if pr.error: return pr
		else:
			stepValue = None
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' before for loop body."))
		self.advance(pr)
		body: list[Expression] = []
		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			# Skip newlines
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail("Expected '}' after for loop body.")
		self.advance(pr)
		return pr.success(ForExpression(symbol, body, startValue, endValue, stepValue))

	def while_loop(self):
		pr = ParseResult()
		self.advance(pr) # Advances past the 'while' keyword
		condition = pr.register(self.expression())
		if pr.error: return pr
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after condition."))
		self.advance(pr)
		body: list[Expression] = []
		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			# Skip newlines
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail("Expected '}' after while loop body.")
		self.advance(pr)
		return pr.success(WhileExpression(condition, body))

	def repeat_loop(self):
		pr = ParseResult()
		self.advance(pr) # Advance past the 'repeat' keyword
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after 'repeat' KEYWORD."))
		self.advance(pr)
		body: list[Expression] = []
		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			# Skip newlines
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail("Expected '}' after repeat loop body.")
		self.advance(pr)
		if self.currentToken != Token(TT.KEYWORD, "until"):
			return pr.fail(InvalidSyntaxError("Expected 'until' KEYWORD after repeat loop body."))
		self.advance(pr)
		condition = pr.register(self.expression())
		if pr.error: return pr
		if self.currentToken.type not in [TT.EOF, TT.NEWLINE]:
			return pr.fail(InvalidSyntaxError("Expected NEWLINE or EOF after repeat condition."))
		return pr.success(RepeatExpression(condition, body))

	def object_declaration(self):
		pr = ParseResult()
		self.advance(pr) # Advances past the 'object' keyword
		if self.currentToken.type != TT.IDENTIFIER:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER (as an object name)."))
		symbol: str = self.currentToken.value
		self.advance(pr)
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after object declaration."))
		self.advance(pr)
		properties: list[Property] = []
		obj = ObjectLiteral(symbol, [])
		while self.currentToken.type == TT.NEWLINE: self.advance(pr) # Skip newlines

		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			
			if self.currentToken.type == TT.IDENTIFIER:
				key: str = self.currentToken.value
				self.advance(pr)

				if self.currentToken.type != TT.COLON:
					return pr.fail(InvalidSyntaxError("Expected ':' after key."))
				self.advance(pr)

				if self.currentToken.type != TT.KEYWORD or self.currentToken.value not in self.dataTypes:
					return pr.fail(InvalidSyntaxError(f"Expected a data type."))
				dataType = self.currentToken.value
				self.advance(pr)
				if dataType == "array":
					if self.currentToken.type != TT.LSQUARE:
						return pr.fail(InvalidSyntaxError("Expected '[' after 'array' keyword."))
					self.advance(pr)
					if self.currentToken.type != TT.KEYWORD or self.currentToken.value not in self.dataTypes[:-1]:
						return pr.fail(InvalidSyntaxError("Expected an array type."))
					dataType += f"[{self.currentToken.value}]"
					self.advance(pr)
					if self.currentToken.type != TT.RSQUARE:
						return pr.fail(InvalidSyntaxError("Expected ']' after array type."))
					self.advance(pr)

				if self.currentToken.type != TT.LARROW:
					return pr.fail(InvalidSyntaxError("Expected '<-' after data type."))
				self.advance(pr)
				value: Expression = pr.register(self.expression())
				if pr.error: return pr
				properties.append(Property(key, dataType, value))
			
				if self.currentToken.type not in [TT.RBRACE, TT.NEWLINE]:
					return pr.fail(InvalidSyntaxError("Expected NEWLINE or '}' after key/value pair."))

			elif self.currentToken == Token(TT.KEYWORD, "method"):
				self.advance(pr)
				method: MethodDeclaration = pr.register(self.method_declaration(obj))
				if pr.error: return pr
				properties.append(Property(method.methodName, "method", method))

			else:
				return pr.fail(InvalidSyntaxError("Expected a key or 'method' keyword."))
			while self.currentToken.type == TT.NEWLINE: self.advance(pr) # Skip newlines
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after object body."))
		self.advance(pr)

		obj.properties = properties
		return pr.success(obj)

	def method_declaration(self, obj: ObjectLiteral):
		pr = ParseResult()
		if self.currentToken.type != TT.IDENTIFIER:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER (as a method name)."))
		methodName = self.currentToken.value
		self.advance(pr)
		if self.currentToken.type != TT.LPAREN:
			return pr.fail(InvalidSyntaxError("Expected '(' after method name."))
		self.advance(pr)
		args: list[tuple[str, str]] = []
		if self.currentToken.type == TT.IDENTIFIER:
			arg = self.currentToken.value
			self.advance(pr)
			if self.currentToken.type != TT.COLON:
				return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
			self.advance(pr)
			if self.currentToken.type != TT.KEYWORD or (self.currentToken.value not in self.dataTypes):
				return pr.fail(InvalidSyntaxError(f"Expected a data type."))
			dataType = self.currentToken.value
			self.advance(pr)
			args.append((arg, dataType))
			while self.currentToken.type == TT.COMMA:
				self.advance(pr)
				if self.currentToken.type != TT.IDENTIFIER:
					return pr.fail(InvalidSyntaxError("Expected IDENTIFIER (as a parameter)."))
				arg = self.currentToken.value
				self.advance(pr)
				if self.currentToken.type != TT.COLON:
					return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
				self.advance(pr)
				if self.currentToken.type != TT.KEYWORD or (self.currentToken.value not in self.dataTypes):
					return pr.fail(InvalidSyntaxError(f"Expected a data type."))
				dataType = self.currentToken.value
				self.advance(pr)
				args.append((arg, dataType))
		# Check for right parenthesis
		if self.currentToken.type != TT.RPAREN:
			return pr.fail(InvalidSyntaxError("Expected IDENTIFIER or ')'."))
		self.advance(pr)
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after method declaration."))
		self.advance(pr)
		body: list[Expression] = []
		while self.currentToken.type not in [TT.EOF, TT.RBRACE]:
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after method body."))
		self.advance(pr)
		return pr.success(MethodDeclaration(methodName, args, body, obj))

	def if_statement(self):
		pr = ParseResult()
		self.advance(pr)
		cases: list[tuple[Expression, list[Expression]]] = []
		elseCase: list[Expression]|None = None
		condition = pr.register(self.expression())
		if pr.error: return pr
		if self.currentToken.type != TT.LBRACE:
			return pr.fail(InvalidSyntaxError("Expected '{' after condition."))
		self.advance(pr)
		body: list[Expression] = []
		while self.currentToken.type not in (TT.EOF, TT.RBRACE):
			stmt = pr.register(self.statement())
			if pr.error: return pr
			body.append(stmt)
			# Skip newlines
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken.type != TT.RBRACE:
			return pr.fail(InvalidSyntaxError("Expected '}' after if body."))
		self.advance(pr)
		cases.append((condition, body))
		while self.currentToken.type == TT.NEWLINE: self.advance(pr)

		# Elif
		while self.currentToken == Token(TT.KEYWORD, "elif"):
			self.advance(pr)
			condition = pr.register(self.expression())
			if pr.error: return pr
			if self.currentToken.type != TT.LBRACE:
				return pr.fail(InvalidSyntaxError("Expected '{' after condition."))
			self.advance(pr)
			body: list[Expression] = []
			while self.currentToken.type not in (TT.EOF, TT.RBRACE):
				stmt = pr.register(self.statement())
				if pr.error: return pr
				body.append(stmt)
				# Skip newlines
				while self.currentToken.type == TT.NEWLINE: self.advance(pr)
			if self.currentToken.type != TT.RBRACE:
				return pr.fail(InvalidSyntaxError("Expected '}' after elif body."))
			self.advance(pr)
			cases.append((condition, body))
			while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		
		# Else
		while self.currentToken.type == TT.NEWLINE: self.advance(pr)
		if self.currentToken == Token(TT.KEYWORD, "else"):
			self.advance(pr)
			if self.currentToken.type != TT.LBRACE:
				return pr.fail(InvalidSyntaxError("Expected '{' after 'else' KEYWORD."))
			self.advance(pr)
			body: list[Expression] = []
			while self.currentToken.type not in (TT.EOF, TT.RBRACE):
				stmt = pr.register(self.statement())
				if pr.error: return pr
				body.append(stmt)
				# Skip newlines
				while self.currentToken.type == TT.NEWLINE: self.advance(pr)
			if self.currentToken.type != TT.RBRACE:
				return pr.fail(InvalidSyntaxError("Expected '}' after else body."))
			self.advance(pr)
			elseCase = body
		return pr.success(IfExpression(cases, elseCase))

	def expression(self):
		pr = ParseResult()

		if self.currentToken == Token(TT.KEYWORD, "inline"):
			# Inline function
			self.advance(pr)
			parameters: list[tuple[str, str]] = []

			if self.currentToken.type == TT.IDENTIFIER:
				parameter = self.currentToken.value
				self.advance(pr)
				
				if self.currentToken.type != TT.COLON:
					return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
				self.advance(pr)

				data_type = self.check_for_data_type()
				if data_type.error: return data_type
				parameters.append((parameter, data_type.node))

				while self.currentToken.type == TT.COMMA:
					self.advance()
					if self.currentToken.type != TT.IDENTIFIER:
						return pr.fail(InvalidSyntaxError("Expected a parameter after ','."))
					parameter = self.currentToken.value
					self.advance(pr)

					if self.currentToken.type != TT.COLON:
						return pr.fail(InvalidSyntaxError("Expected ':' after parameter."))
					self.advance(pr)

					data_type = self.check_for_data_type()
					if data_type.error: return data_type
					parameters.append((parameter, data_type.node))
			
			if self.currentToken.type != TT.RARROW:
				return pr.fail(InvalidSyntaxError("Expected '->' after inline function declaration."))
			self.advance(pr)

			if self.currentToken.type != TT.LBRACE:
				return pr.fail(InvalidSyntaxError("Expected '{' before inline function definition."))
			self.advance(pr)

			body = pr.register(self.statement())
			if pr.error: return pr
			
			if self.currentToken.type != TT.RBRACE:
				return pr.fail(InvalidSyntaxError("Expected '}' after inline function definition."))
			self.advance(pr)
			return pr.success(FunctionDeclaration("<inline>", parameters, [body], True))

		condition = pr.register(self.assignment())
		if pr.error: return pr
		if self.currentToken.type == TT.QUESTION:
			self.advance(pr)
			truthyValue = pr.register(self.expression())
			if pr.error: return pr
			if self.currentToken.type != TT.COLON:
				return pr.fail(InvalidSyntaxError("Expected ':' after truthy value."))
			self.advance(pr)
			falsyValue = pr.register(self.expression())
			if pr.error: return pr
			return pr.success(TernaryOperation(condition, truthyValue, falsyValue))
		return pr.success(condition)
	
	def assignment(self):
		pr = ParseResult()
		literal = pr.register(self.logical())
		if pr.error: return pr

		if isinstance(literal, Identifier):
			symbol: str = literal.value
			if not (self.currentToken.type == TT.OPERATION and self.currentToken.value in ("=", "+=", "-=", "*=", "/=", "%=", "^=")):
				self.step_back()
				return self.logical()
			# Assignment
			operation = self.currentToken.value
			self.advance(pr)
			value = pr.register(self.expression())
			if pr.error: return pr
			return pr.success(Assignment(symbol, value, operation))
		
		return pr.success(literal)
	
	def logical(self):
		pr = ParseResult()
		left = pr.register(self.comparison())
		if pr.error: return pr

		while self.currentToken in (Token(TT.KEYWORD, "and"), Token(TT.KEYWORD, "or")):
			operator = self.currentToken
			self.advance(pr)
			right = pr.register(self.comparison())
			if pr.error: return pr
			left = BinaryOperation(left, operator, right)
		return pr.success(left)

	def comparison(self):
		pr = ParseResult()
		left = pr.register(self.bitwise())
		if pr.error: return pr
		while self.currentToken.type == TT.OPERATION and self.currentToken.value in (">", ">=", "==", "!=", "<=", "<"):
			operator = self.currentToken
			self.advance(pr)
			right = pr.register(self.bitwise())
			if pr.error: return pr
			left = BinaryOperation(left, operator, right)
		return pr.success(left)
	
	def bitwise(self):
		pr = ParseResult()
		left = pr.register(self.additive())
		if pr.error: return pr
		while self.currentToken.type == TT.OPERATION and self.currentToken.value in ("<<", ">>"):
			operator = self.currentToken
			self.advance(pr)
			right = pr.register(self.additive())
			if pr.error: return pr
			left = BinaryOperation(left, operator, right)
		return pr.success(left)

	def additive(self):
		pr = ParseResult()
		left = pr.register(self.multiplicative())
		if pr.error: return pr
		while self.currentToken.type == TT.OPERATION and self.currentToken.value in "+-":
			operator = self.currentToken
			self.advance(pr)
			right = pr.register(self.multiplicative())
			if pr.error: return pr
			left = BinaryOperation(left, operator, right)
		return pr.success(left)
	
	def multiplicative(self):
		pr = ParseResult()
		left = pr.register(self.unary())
		if pr.error: return pr
		while self.currentToken.type == TT.OPERATION and self.currentToken.value in "*/%":
			operator = self.currentToken
			self.advance(pr)
			right = pr.register(self.unary())
			if pr.error: return pr
			left = BinaryOperation(left, operator, right)
		return pr.success(left)
	
	def unary(self):
		pr = ParseResult()
		if self.currentToken in (Token(TT.OPERATION, "+"), Token(TT.OPERATION, "-"), Token(TT.KEYWORD, "not")):
			operator = self.currentToken
			self.advance(pr)
			expr = pr.register(self.unary())
			if pr.error: return pr
			return pr.success(UnaryOperation(operator, expr))
		return self.power()
	
	def power(self):
		pr = ParseResult()
		left = pr.register(self.call_expression())
		if pr.error: return pr
		if self.currentToken == Token(TT.OPERATION, "^"):
			while self.currentToken == Token(TT.OPERATION, "^"):
				self.advance(pr)
				right = pr.register(self.unary())
				if pr.error: return pr
				left = BinaryOperation(left, Token(TT.OPERATION, "^"), right)
		return pr.success(left)
	
	def call_expression(self):
		pr = ParseResult()
		literal = pr.register(self.member_expression())
		if pr.error: return pr

		if self.currentToken.type == TT.LPAREN: # Call function/method
			if not isinstance(literal, (Identifier, MemberExpression)):
				return pr.fail(InvalidSyntaxError("Expected identifier (as a function name)."))
			self.advance(pr)
			args: list[Expression] = []
			if self.currentToken.type != TT.RPAREN:
				# First argument
				arg = pr.register(self.expression())
				if pr.error: return pr
				args.append(arg)
				while self.currentToken.type == TT.COMMA:
					self.advance(pr)
					arg = pr.register(self.expression())
					if pr.error: return pr
					args.append(arg)
				if self.currentToken.type != TT.RPAREN:
					return pr.fail(InvalidSyntaxError("Expected ',' or ')'."))
			self.advance(pr)
			return pr.success(CallExpression(literal, args))	
		elif self.currentToken.type == TT.LSQUARE: # Access value in an array
			if not isinstance(literal, (Identifier, MemberExpression, ArrayLiteral, StringLiteral)):
				return pr.fail(InvalidSyntaxError("Expected identifier, string, or array."))
			self.advance(pr)
			index: Expression = pr.register(self.expression())
			end_index = None
			if pr.error: return pr

			if self.currentToken.type == TT.COLON:
				self.advance()
				end_index = pr.register(self.expression())
				if pr.error: return pr

			if self.currentToken.type != TT.RSQUARE:
				return pr.fail(InvalidSyntaxError("Expected ']' after array access expression"))
			self.advance(pr)
			if not (self.currentToken.type == TT.OPERATION and self.currentToken.value in ("=", "+=", "-=", "*=", "/=", "%=", "^=")): return pr.success(ArrayAccessExpression(literal, index, end_index))

			# Set value
			operation = self.currentToken.value
			self.advance(pr)
			value = pr.register(self.expression())
			if pr.error: return pr
			return pr.success(ArraySetExpression(literal, index, operation, value))
		else: return pr.success(literal)

	def member_expression(self):
		pr = ParseResult()
		obj = pr.register(self.literal())
		if pr.error: return pr
		
		if self.currentToken.type == TT.DOT:
			if not isinstance(obj, Identifier):
				return pr.fail(InvalidSyntaxError("Expected an object identifier."))
			while self.currentToken.type == TT.DOT:
				self.advance(pr)
				_property: Expression = pr.register(self.literal())
				if pr.error: return pr
				if not isinstance(_property, Identifier):
					return pr.fail(InvalidSyntaxError("Expected a property/method."))
			
				obj = MemberExpression(obj, _property)
		
		return pr.success(obj)

	def literal(self):
		pr = ParseResult()
		tok = self.currentToken
		self.advance(pr)
		if tok.type == TT.INTEGER:
			return pr.success(IntLiteral(tok.value))
		elif tok.type == TT.REAL:
			return pr.success(RealLiteral(tok.value))
		elif tok.type == TT.IDENTIFIER:
			return pr.success(Identifier(tok.value))
		elif tok.type == TT.STRING:
			return pr.success(StringLiteral(tok.value))
		elif tok.type == TT.CHAR:
			return pr.success(CharLiteral(tok.value))
		elif tok.type == TT.LPAREN:
			expr = pr.register(self.expression())
			if pr.error: return pr
			# Check for right paranthesis
			if self.currentToken.type != TT.RPAREN:
				return pr.fail(InvalidSyntaxError("Expected ')'."))
			self.advance(pr)
			return pr.success(expr)
		elif tok.type == TT.LSQUARE:
			values: list[Expression] = []
			
			# Blank array
			if self.currentToken.type == TT.RSQUARE:
				self.advance(pr)
				return pr.success(ArrayLiteral(values))
			
			# First value
			value: Expression = pr.register(self.expression())
			if pr.error: return pr
			values.append(value)
			dataType = type(value).__name__

			# Multiple values
			while self.currentToken.type == TT.COMMA:
				self.advance(pr)
				value = pr.register(self.expression())
				if pr.error: return pr
				if type(value).__name__ != dataType:
					if dataType == "RealLiteral" and isinstance(value, IntLiteral):
						value = RealLiteral(float(value.value))
					elif dataType == "IntLiteral" and isinstance(value, RealLiteral):
						dataType = "RealLiteral"

						# Convert to an array of reals
						for i in range(len(values)):
							values[i] = RealLiteral(float(values[i].value))
					else:
						return pr.fail(InvalidSyntaxError(f"Expected {dataType}, found {type(value).__name__} instead."))

				values.append(value)

			if self.currentToken.type != TT.RSQUARE:
				return pr.fail(InvalidSyntaxError("Expected ',' or ']'."))
			self.advance(pr)
			return pr.success(ArrayLiteral(values))
		else:
			print(self.tokenIndex - 1)
			return pr.fail(InvalidSyntaxError("Expected an integer, real, identifier, string, char, '[', or '('."))

## RUNTIME VALUE
class RTValue:
	def __init__(self):
		self.value = None

	def illegalOperation(self, other: Self|None = None):
		if other:
			return None, RTError(f"Illegal operation between {type(self).__name__.lower()} and {type(other).__name__.lower()}.")
		return None, RTError(f"Illegal operation for {type(self).__name__.lower()}.")
	
	def __add__(self, other: Self): return self.illegalOperation(other)
	def __sub__(self, other: Self): return self.illegalOperation(other)
	def __mul__(self, other: Self): return self.illegalOperation(other)
	def __truediv__(self, other: Self): return self.illegalOperation(other)
	def __mod__(self, other: Self): return self.illegalOperation(other)
	def __pow__(self, other: Self): return self.illegalOperation(other)
	def __lt__(self, other: Self): return self.illegalOperation(other)
	def __le__(self, other: Self): return self.illegalOperation(other)
	def __eq__(self, other: Self): return self.illegalOperation(other)
	def __ne__(self, other: Self): return self.illegalOperation(other)
	def __gt__(self, other: Self): return self.illegalOperation(other)
	def __ge__(self, other: Self): return self.illegalOperation(other)
	def __call__(self, args: list[Expression], output): return self.illegalOperation()
	def __bool__(self): return False
	def __len__(self): return self.illegalOperation()
	def __lshift__(self, other: Self): return self.illegalOperation(other)
	def __rshift__(self, other: Self): return self.illegalOperation(other)
	def __and__(self, other: Self): return self.illegalOperation(other)
	def __or__(self, other: Self): return self.illegalOperation(other)
	def __invert__(self): return self.illegalOperation()

class Null(RTValue):
	def __init__(self):
		self.value = None

	def __repr__(self):
		return "<null>"
	
	def __bool__(self):
		return False

class Integer(RTValue):
	def __init__(self, value: int):
		self.value = value

	def __repr__(self):
		return str(self.value)
	
	def __bool__(self):
		return self.value != 0
	
	def __int__(self):
		return self.value
	
	def __add__(self, other: RTValue):
		if isinstance(other, Integer):
			return Integer(self.value + other.value), None
		elif isinstance(other, Real):
			return Real(self.value + other.value), None
		return self.illegalOperation(other)
	
	def __sub__(self, other: RTValue):
		if isinstance(other, Integer):
			return Integer(self.value - other.value), None
		elif isinstance(other, Real):
			return Real(self.value - other.value), None
		return self.illegalOperation(other)
	
	def __mul__(self, other: RTValue):
		if isinstance(other, Integer):
			return Integer(self.value * other.value), None
		elif isinstance(other, Real):
			return Real(self.value * other.value), None
		return self.illegalOperation(other)
	
	def __truediv__(self, other: RTValue):
		if isinstance(other, Integer):
			if other.value == 0:
				return None, RTError("Division by 0.")
			return Integer(self.value // other.value), None # Integer division
		elif isinstance(other, Real):
			if other.value == 0.0:
				return None, RTError("Division by 0.")
			return Real(self.value / other.value), None
		return self.illegalOperation(other)
	
	def __mod__(self, other: RTValue):
		if isinstance(other, Integer):
			if other.value == 0:
				return None, RTError("Modulus by 0.")
			return Integer(self.value % other.value), None
		if isinstance(other, Real):
			if other.value == 0.0:
				return None, RTError("Modulus by 0.")
			return Integer(self.value % other.value), None
		return self.illegalOperation(other)
	
	def __lt__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value < other.value), None
		return self.illegalOperation(other)
	
	def __le__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value <= other.value), None
		return self.illegalOperation(other)
	
	def __eq__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value == other.value), None
		return self.illegalOperation(other)
	
	def __ne__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value != other.value), None
		return self.illegalOperation(other)
	
	def __gt__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value > other.value), None
		return self.illegalOperation(other)
	
	def __ge__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value >= other.value), None
		return self.illegalOperation(other)
	
	def __pow__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			res = (self.value ** other.value)
			if res == int(res):
				return Integer(res), None
			return Real(res), None
		return self.illegalOperation(other)

	def __lshift__(self, other: RTValue):
		if isinstance(other, Integer):
			return Integer(self.value << other.value), None
		return self.illegalOperation(other)
	
	def __rshift__(self, other: RTValue):
		if isinstance(other, Integer):
			return Integer(self.value >> other.value), None
		return self.illegalOperation(other)

class Real(RTValue):
	def __init__(self, value: float):
		self.value = value
	
	def __repr__(self):
		return str(self.value)

	def __bool__(self):
		return self.value != 0.0
	
	def __add__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Real(self.value + other.value), None
		return self.illegalOperation(other)
	
	def __sub__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Real(self.value - other.value), None
		return self.illegalOperation(other)
	
	def __mul__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Real(self.value * other.value), None
		return self.illegalOperation(other)
	
	def __truediv__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			if other.value == 0:
				return None, RTError("Division by 0.")
			return Real(self.value / other.value), None
		return self.illegalOperation(other)
	
	def __mod__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			if other.value == 0:
				return None, RTError("Modulus by 0.")
			return Real(self.value % other.value), None
		return self.illegalOperation(other)
	
	def __lt__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value < other.value), None
		return self.illegalOperation(other)
	
	def __le__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value <= other.value), None
		return self.illegalOperation(other)
	
	def __eq__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value == other.value), None
		return self.illegalOperation(other)
	
	def __ne__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value != other.value), None
		return self.illegalOperation(other)
	
	def __gt__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value > other.value), None
		return self.illegalOperation(other)
	
	def __ge__(self, other: RTValue):
		if isinstance(other, (Integer, Real)):
			return Boolean(self.value >= other.value), None
		return self.illegalOperation(other)
	
	def __pow__(self, other):
		if isinstance(other, (Integer, Real)):
			res = (self.value ** other.value)
			if res == int(res):
				return Integer(res), None
			return Real(res), None
		return self.illegalOperation(other)

class String(RTValue):
	def __init__(self, value: str):
		self.value = value
	
	def __repr__(self):
		return f"\"{self.value}\""

	def __bool__(self):
		return len(self.value) > 0
	
	def __str__(self):
		return self.value
	
	def __add__(self, other: RTValue):
		if isinstance(other, (String, Char)):
			return String(self.value + other.value), None
		return self.illegalOperation(other)
	
	def __mul__(self, other: RTValue):
		if isinstance(other, Integer):
			return String(self.value * other.value), None
		return self.illegalOperation(other)

	def __len__(self):
		return Integer(len(self.value))

	def __eq__(self, other: RTValue):
		if isinstance(other, String):
			return Boolean(self.value == other.value), None
		return self.illegalOperation(other)
	
	def __ne__(self, other: RTValue):
		if isinstance(other, String):
			return Boolean(self.value != other.value), None
		return self.illegalOperation(other)

class Char(RTValue):
	def __init__(self, value: str = ''):
		self.value = value

	def __repr__(self):
		return f"'{self.value}'"
	
	def __bool__(self):
		return len(self.value) == 1
	
	def __str__(self):
		return self.value

	def __add__(self, other: RTValue):
		if isinstance(other, Char):
			result = ord(self.value) + ord(other.value)
			if result < 0 or result > 0x10ffff:
				return None, RTError(f"Cannot retrieve character {hex(result)}.")
			return Char(chr(result)), None
		return self.illegalOperation(other)
	
	def __sub__(self, other: RTValue):
		if isinstance(other, Char):
			result = ord(self.value) - ord(other.value)
			if result < 0 or result > 0x10ffff:
				return None, RTError(f"Cannot retrieve character {hex(result)}.")
			return Char(chr(result)), None
		return self.illegalOperation(other)

	def __lshift__(self, other: RTValue):
		if isinstance(other, Integer):
			result = ord(self.value) << other.value
		elif isinstance(other, Char):
			result = ord(self.value) << ord(other.value)
		else:
			return self.illegalOperation(other)
		if result < 0 or result > 0x10ffff:
			return None, RTError(f"Cannot retrieve character {hex(result)}.")
		return Char(chr(result)), None
	
	def __rshift__(self, other: RTValue):
		if isinstance(other, Integer):
			result = ord(self.value) >> other.value
		elif isinstance(other, Char):
			result = ord(self.value) >> ord(other.value)
		else:
			return self.illegalOperation(other)
		if result < 0 or result > 0x10ffff:
			return None, RTError(f"Cannot retrieve character {hex(result)}.")
		return Char(chr(result)), None

class Boolean(RTValue):
	def __init__(self, value: bool):
		self.value = value
	
	def __repr__(self):
		return f"{self.value}".lower()
	
	def __bool__(self):
		return self.value

	def __and__(self, other: RTValue):
		if isinstance(other, Boolean):
			return Boolean(self.value and other.value), None
		return self.illegalOperation(other)
	
	def __or__(self, other: RTValue):
		if isinstance(other, Boolean):
			return Boolean(self.value or other.value), None
		return self.illegalOperation(other)

	def __invert__(self):
		return Boolean(not self.value), None

class Object(RTValue):
	def __init__(self, symbol: str, properties: dict[str, RTValue], env):
		self.symbol = symbol
		self.properties = properties
		self.scope = Environment(env)
		self.scope.symbols = properties
	
	def __repr__(self):
		return f"object <{self.symbol}> " + "{\n" + "\n".join([f"\t{key}: {repr(self.properties[key])}" for key in self.properties.keys()]) + "\n}"

class Function(RTValue):
	def __init__(self, name: str, args: list[tuple[str, str]], body: list[Expression], env, should_auto_return: bool, argNum: int|None = None):
		self.name = name
		self.args = args
		self.body = body
		self.argNum = argNum
		self.should_auto_return = should_auto_return
		self.scope = Environment(env)
		# Declare args in scope
		for arg in args:
			match arg[1]:
				case "integer": value = Integer(0)
				case "real": value = Real(0.0)
				case "boolean": value = Boolean(False)
				case "string": value = String("")
				case "char": value = Char()
				case "array[integer]": value = Array([], "integer")
				case "array[real]": value = Array([], "real")
				case "array[boolean]": value = Array([], "boolean")
				case "array[string]": value = Array([], "string")
				case "array[char]": value = Array([], "char")
				case "file": value = FileHandler("")
				case _: value = Null()
			self.scope.declare(arg[0], False, value)
	
	def __repr__(self):
		return f"<function {self.name}>"
	
	def __call__(self, args: list[Expression], output):
		res = RTResult()
		# Check arguments
		if self.argNum != None: # Fixed number of arguments
			if len(args) > self.argNum:
				return res.fail(RTError(f"{len(args) - self.argNum} too many arguments for function {self.name}()."))
			if len(args) < self.argNum:
				return res.fail(RTError(f"{self.argNum - len(args)} too few arguments for function {self.name}()."))
		
		# Populate arguments
		interpreter = Interpreter()
		for i, arg in enumerate(args):
			argValue = res.register(interpreter.visit(arg, self.scope, output))
			if res.should_return(): return res
			if self.argNum != None:
				self.scope.set(self.args[i][0], argValue)
			else:
				self.scope.declare(f"arg_{i}", False, argValue)
		
		# Execute body line by line
		result = res.register(interpreter.visit(Program(self.body), self.scope, output))
		if res.should_return() and isinstance(res.return_value, Null): return res
		self.scope.symbols = dict()
		# Declare args in scope
		for arg in self.args:
			match arg[1]:
				case "integer": value = Integer(0)
				case "real": value = Real(0.0)
				case "boolean": value = Boolean(False)
				case "string": value = String("")
				case "char": value = Char()
				case "array[integer]": value = Array([], "integer")
				case "array[real]": value = Array([], "real")
				case "array[boolean]": value = Array([], "boolean")
				case "array[string]": value = Array([], "string")
				case "array[char]": value = Array([], "char")
				case "file": value = FileHandler("")
				case _: value = Null()
			self.scope.declare(arg[0], False, value)
		
		if not isinstance(res.return_value, Null):
			return_value = res.return_value
		elif self.should_auto_return:
			return_value = result
		else:
			return_value = Null()

		return res.success(return_value)

class NativeFunction(Function):
	def __init__(self, name: str, args: list[tuple[str, str]], env, argNum: int|None = None):
		self.name = name
		self.args = args
		self.argNum = argNum
		self.env = env
		self.scope = Environment(env)
		# Declare args in scope
		for arg in args:
			match arg[1]:
				case "integer": value = Integer(0)
				case "real": value = Real(0.0)
				case "boolean": value = Boolean(False)
				case "string": value = String("")
				case "char": value = Char()
				case "array[integer]": value = Array([], "integer")
				case "array[real]": value = Array([], "real")
				case "array[boolean]": value = Array([], "boolean")
				case "array[string]": value = Array([], "string")
				case "array[char]": value = Array([], "char")
				case "file": value = FileHandler("")
				case _: value = Null()
			self.scope.declare(arg[0], False, value)
		
	def __repr__(self):
		return f"<nativefunction {self.name.lower()}>"
	
	def __call__(self, args: list[Expression], output):
		res = RTResult()
		# Check arguments
		if self.argNum != None: # Fixed number of arguments
			if len(args) > self.argNum:
				return res.fail(RTError(f"{len(args) - self.argNum} too many arguments for function {self.name}()."))
			if len(args) < self.argNum:
				return res.fail(RTError(f"{self.argNum - len(args)} too few arguments for function {self.name}()."))
		
		# Populate arguments
		interpreter = Interpreter()
		if self.argNum != None:
			for i, arg in enumerate(args):
				argValue = res.register(interpreter.visit(arg, self.scope, output))
				if res.should_return(): return res
				self.scope.set(self.args[i][0], argValue)
		else:
			for i, arg in enumerate(args):
				argValue = res.register(interpreter.visit(arg, self.scope, output))
				if res.should_return(): return res
				self.scope.declare(f"arg_{i}", False, argValue)
		
		# Execute body line by line
		func = getattr(self, f"call{self.name}")
		result = res.register(func(output))
		if res.should_return(): return res

		if self.argNum == None: self.scope.symbols = dict()
		return res.success(result)

	def callPrint(self, output):
		try:
			printValues: str = "".join([f"{self.scope.symbols.get(arg)}" for arg in self.scope.symbols])
		except ValueError as e:
			return RTResult().fail(RTError(f"{e}"))
		output.terminalEdit.append(printValues)
		return RTResult().success(Null())
	
	def callRead(self, output):
		global input_pos
		res = RTResult()
		output.terminalEdit.append("> ")
		input_pos = len(output.terminalEdit.toPlainText())
		output.terminalEdit.setReadOnly(False)

		output.loop.exec() # Waits for user input

		return res.success(String(output.terminalEdit.toPlainText()[input_pos:]))

	def callError(self, output):
		res = RTResult()
		message: String = res.register(self.scope.get("message"))
		if res.should_return(): return res
		return res.fail(RTError(message.value))

	def callToString(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, String):
			return res.success(value)
		if isinstance(value, (Integer, Real, Char, Typing, Array)):
			return res.success(String(str(value.value)))
		if isinstance(value, Boolean):
			return res.success(String(str(value.value).lower()))
		return res.fail(RTError(f"Unable to convert '{value}' to a string."))
	
	def callToInteger(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, (Integer, Real, Boolean)):
			return res.success(Integer(int(value.value)))
		if isinstance(value, Char):
			return res.success(Integer(ord(value.value)))
		if isinstance(value, String):
			try:
				return_value = int(value.value)
			except:
				return res.fail(RTError(f"Unable to convert '{value}' to an integer."))
			else:
				return res.success(Integer(return_value))
		return res.fail(RTError(f"Unable to convert '{value}' to an integer."))
	
	def callToReal(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, (Integer, Real, Boolean)):
			return res.success(Real(float(value.value)))
		if isinstance(value, Char):
			return res.success(Real(float(ord(value.value))))
		if isinstance(value, String):
			try:
				return_value = float(value.value)
			except:
				return res.fail(RTError(f"Unable to convert '{value}' to a real."))
			else:
				return res.success(Real(return_value))
		return res.fail(RTError(f"Unable to convert '{value}' to a real."))
	
	def callToBoolean(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, (Integer, Real, Boolean)):
			return res.success(Boolean(bool(value.value)))
		if isinstance(value, Char):
			return res.success(Boolean(bool(ord(value.value))))
		if isinstance(value, String):
			return res.success(Boolean(len(value.value) > 0))
		return res.fail(RTError(f"Unable to convert '{value}' to a boolean."))

	def callToChar(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, Integer):
			return res.success(Char(chr(value.value)))
		if isinstance(value, Char):
			return res.success(value)
		if isinstance(value, String):
			if len(value.value) > 1:
				return res.fail(RTError(f"Unable to convert '{value}' to a character."))
			if len(value.value) == 0:
				return res.success(Char('\0'))
			return res.success(Char(value.value))
		return res.fail(RTError(f"Unable to convert '{value}' to a character."))

	def callIsString(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.error: return res
		return res.success(Boolean(isinstance(value, String)))
	
	def callIsInteger(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.error: return res
		return res.success(Boolean(isinstance(value, Integer)))
	
	def callIsReal(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.error: return res
		return res.success(Boolean(isinstance(value, Real)))
	
	def callIsBoolean(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.error: return res
		return res.success(Boolean(isinstance(value, Boolean)))
	
	def callIsChar(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.error: return res
		return res.success(Boolean(isinstance(value, Char)))

	def callFormat(self, output):
		res = RTResult()
		values: list[RTValue] = []
		for key in self.scope.symbols.keys():
			values.append(self.scope.symbols[key])
		toFormat: str = values[0].value
		if not isinstance(toFormat, str):
			return res.fail("Argument 'str' must be a string.")
		if toFormat.count("{}") != len(values) - 1:
			return res.fail("Cannot format argument 'str'.")
		count: int = toFormat.count("{}")
		replacements: int = toFormat.count("{}")
		while replacements > 0:
			replacements -= 1
			toFormat = toFormat.replace("{}", f"{values[count - replacements]}", 1)
		return res.success(String(toFormat))

	def callPow(self, output):
		"""
		pow(a, b) -> a ^ b
		pow(a, b, c) -> a ^ b % c
		"""
		res = RTResult()
		values: list[RTValue] = []
		for i, key in enumerate(self.scope.symbols.keys()):
			value = self.scope.symbols[key]
			if i < 2:
				if not isinstance(value, (Integer, Real)):
					return res.fail(RTError(f"Argument {i+1} must be an integer/real."))
			elif i > 2:
				return res.fail(RTError("Expected 2/3 arguments for pow() function."))
			else:
				if not isinstance(value, (Integer)):
					return res.fail(RTError("Argument 3 must be an integer."))
			values.append(value)
		if len(values) == 2:
			power = pow(values[0].value, values[1].value)
			return res.success(Real(power) if isinstance(power, float) else Integer(power))
		try:
			power = pow(values[0].value, values[1].value, values[2].value)
		except:
			return res.fail(RTError("Argument 3 of pow() function cannot be 0."))
		return res.success(Real(power) if isinstance(power, float) else Integer(power))

	def callType(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		return res.success(Typing(type(value).__name__.lower()))

	def callLength(self, output):
		res = RTResult()
		arg = self.scope.get("value")
		if arg.error: return arg
		value = arg.value

		if isinstance(value, (String, Array)):
			return res.success(Integer(len(value.value)))
		return res.fail(RTError(f"Cannot retrieve the length of {type(value).__name__.lower()}."))

	def callReverse(self, output):
		res = RTResult()
		value = res.register(self.scope.get("value"))
		if res.should_return(): return res
		if not isinstance(value, (String, Array)):
			return res.fail(RTError("Expected a string or an array for argument 'value'."))
		if isinstance(value, String):
			return res.success(String(value.value[::-1]))
		return res.success(Array(value.value[::-1]))

	def callSorted(self, output):
		res = RTResult()
		array: Array = res.register(self.scope.get("arr"))
		if res.should_return(): return res
		array_value = array.value

		def get_value(val: RTValue):
			return val.value
		
		return res.success(Array(sorted(array_value, key=get_value)))
	
	def callSum(self, output):
		res = RTResult()
		array: Array = res.register(self.scope.get("arr"))
		if res.should_return(): return res
		
		if array.arrayType not in ("integer", "real", "char"):
			return res.fail(RTError(f"Cannot sum elements of an array[{array.arrayType}]."))
		
		match array.arrayType:
			case "integer":
				result: int = 0
				for item in array.value:
					result += item.value
				return res.success(Integer(result))
			case "real":
				result: float = 0.0
				for item in array.value:
					result += item.value
				return res.success(Real(result))
			case "char":
				result: int = 0
				for item in array.value:
					result += ord(item.value)
				
				if result < 0 or result > 0x10ffff:
					return res.fail(RTError(f"Cannot retrieve character #{hex(result)}."))
				return res.success(Char(chr(result)))

	def callOpenfile(self, output):
		res = RTResult()
		fn_arg = self.scope.get("fn")
		if fn_arg.error: return fn_arg
		fn: String = fn_arg.value

		mode_arg = self.scope.get("mode")
		if mode_arg.error: return mode_arg
		if not isinstance(mode_arg.value, Char):
			return res.fail(RTError("Argument 'mode' must be a character."))
		mode: Char = mode_arg.value

		if mode.value not in "rwa":
			return res.fail(RTError("Expected 'r', 'w', or 'a' for argument 'mode'."))

		try:
			file = open(f"Projects/{user}/{fn.value}", mode.value)
		except:
			return res.fail(RTError(f"File '{fn.value}' does not exist."))
		
		result = FileHandler(fn.value, file)
		return res.success(result)
	
	def callClosefile(self, output):
		res = RTResult()
		file_res = self.scope.get("file")
		if file_res.error: return file_res
		file: FileHandler = file_res.value

		try:
			file.file.close()
		except:
			return res.fail(RTError(f"File '{file.fn}' is unopened."))
		return res.success(Null())

	def callReadfile(self, output):
		res = RTResult()
		file_res = self.scope.get("file")
		if file_res.error: return file_res
		file: FileHandler = file_res.value

		if not file.file.readable():
			return res.fail(RTError(f"Missing read permission to read from file '{file.fn}'."))
		line: str = file.file.readline()
		if line == "":
			file.eof = True
		return res.success(String(line))

	def callWritefile(self, output):
		res = RTResult()
		file_res = self.scope.get("file")
		if file_res.error: return file_res
		file: FileHandler = file_res.value

		content_res = self.scope.get("content")
		if content_res.error: return content_res
		content: String = content_res.value

		if not file.file.writable():
			return res.fail(RTError(f"Missing write permission to write to file '{file.fn}'."))
		file.file.write(content.value)
		return res.success(Null())

	def callIseof(self, output):
		res = RTResult()
		file_res = self.scope.get("file")
		if file_res.error: return file_res
		file: FileHandler = file_res.value

		return res.success(Boolean(file.eof))

	def callMax(self, output):
		res = RTResult()
		x = res.register(self.scope.get("x"))
		if res.should_return(): return res
		y = res.register(self.scope.get("y"))
		if res.should_return(): return res

		if type(x).__name__ != type(y).__name__:
			if (type(x).__name__ + type(y).__name__) not in ("IntegerReal", "RealInteger"):
				return res.fail(RTError(f"Cannot find the maximum between type <{type(x).__name__.lower()}> and type <{type(y).__name__.lower()}>."))
		
		try:
			if isinstance(x, (Boolean, Object, Array, FileHandler, Function, NativeFunction)):
				raise Exception
			if isinstance(y, (Boolean, Object, Array, FileHandler, Function, NativeFunction)):
				raise Exception
			if isinstance(x, Char):
				result = Char(x.value if ord(x.value) > ord(y.value) else y.value)
			else:
				result = x if x.value > y.value else y
		except:
			return res.fail(RTError(f"Cannot find the maximum between type <{type(x).__name__.lower()}> and type <{type(y).__name__.lower()}>."))
		else:
			return res.success(result)
		
	def callMin(self, output):
		res = RTResult()
		x = res.register(self.scope.get("x"))
		if res.should_return(): return res
		y = res.register(self.scope.get("y"))
		if res.should_return(): return res

		if type(x).__name__ != type(y).__name__:
			if (type(x).__name__ + type(y).__name__) not in ("IntegerReal", "RealInteger"):
				return res.fail(RTError(f"Cannot find the minimum between type <{type(x).__name__.lower()}> and type <{type(y).__name__.lower()}>."))
		
		try:
			if isinstance(x, (Boolean, Object, Array, FileHandler, Function, NativeFunction)):
				raise Exception
			if isinstance(y, (Boolean, Object, Array, FileHandler, Function, NativeFunction)):
				raise Exception
			if isinstance(x, Char):
				result = Char(x.value if ord(x.value) < ord(y.value) else y.value)
			else:
				result = x if x.value < y.value else y
		except:
			return res.fail(RTError(f"Cannot find the minimum between type <{type(x).__name__.lower()}> and type <{type(y).__name__.lower()}>."))
		else:
			return res.success(result)

	def callRand(self, output):
		return RTResult().success(Real(random()))

	def callRandint(self, output):
		res = RTResult()
		start: Integer = res.register(self.scope.get("start"))
		if res.should_return(): return res
		end: Integer = res.register(self.scope.get("end"))
		if res.should_return(): return res

		return res.success(Integer(randint(start.value, end.value)))

class Method(Function):
	def __init__(self, name: str, args: list[tuple[str, str]], body: list[Expression], obj: Object, argNum: int|None = None):
		self.name = name
		self.args = args
		self.body = body
		self.argNum = argNum
		self.obj = obj
		self.scope = Environment(obj.scope)
		self.should_auto_return = False
		# Declare args in scope
		for arg in args:
			match arg[1]:
				case "integer": value = Integer(0)
				case "real": value = Real(0.0)
				case "boolean": value = Boolean(False)
				case "string": value = String("")
				case _: value = Null()
			self.scope.declare(arg[0], False, value)
	
	def __repr__(self):
		return f"<method {self.name}>"

class Array(RTValue):
	def __init__(self, value: list[RTValue], arrayType: str = ""):
		self.value = value
		if arrayType == "" and len(value) > 0:
			self.arrayType = type(value[0]).__name__.lower()
		else:
			self.arrayType = arrayType.lower()
	
	def __repr__(self):
		return f"{self.value}"
	
	def copy(self):
		copy = Array(self.value, self.arrayType)
		return copy
		
	def illegalOperation(self, other: RTValue|None = None):
		if isinstance(other, Array):
			return None, RTError(f"Illegal operation between array[{self.arrayType.lower()}] and array[{other.arrayType.lower()}].")
		elif other:
			return None, RTError(f"Illegal operation between array[{self.arrayType.lower()}] and {type(other).__name__.lower()}.")
		return None, RTError(f"Illegal operation for array[{self.arrayType.lower()}]")
		
	def illegalAppend(self, other: Self):
		return None, RTError(f"Illegal appending between array[{self.arrayType.lower()}] and array[{other.arrayType.lower()}].")

	def __len__(self):
		return Integer(len(self.value))

	def __add__(self, other: RTValue):
		if isinstance(other, Array):
			if self.arrayType == "":
				self.arrayType = other.arrayType
			elif self.arrayType != other.arrayType:
				return self.illegalAppend(other)
			
			return_value = self.value.copy()
			return_value.extend(other.value.copy())
			return Array(return_value, self.arrayType), None
		return self.illegalOperation(other)

	def __mul__(self, other: RTValue):
		if isinstance(other, Integer):
			if other.value <= 0:
				return None, RTError("Array can only be multiplied by a positive integer.")
			return Array(self.value * other.value), None
		return self.illegalOperation(other)

class Typing(RTValue):
	def __init__(self, value: str):
		self.value = value

	def __repr__(self):
		return f"type <{self.value}>"

class FileHandler(RTValue):
	def __init__(self, fn: str, file: IO|None = None):
		self.fn = fn
		self.file = file
		self.eof = False
	
	def __repr__(self):
		return f"filehandler <{self.fn}>"
	
	def __bool__(self):
		return len(self.fn) > 0

## ENVIRONMENT
class Environment:
	def __init__(self, parent: Self|None = None):
		self.parent = parent
		self.symbols: dict[str, RTValue] = {}
		self.constants: list[str] = []
	
	def declare(self, symbol: str, isConst: bool, value: RTValue):
		res = RTResult()
		if self.symbols.get(symbol, None) != None:
			return res.fail(RTError(f"Cannot redeclare the symbol '{symbol}'."))
		self.symbols[symbol] = value
		if isConst: self.constants.append(symbol)
		return res.success(Null())
	
	def get(self, symbol: str):
		res = RTResult()
		if symbol in self.symbols:
			value = self.symbols[symbol]
		else:
			if self.parent != None: 
				return self.parent.get(symbol)
			else:
				return res.fail(RTError(f"Symbol '{symbol}' does not exist."))
		return res.success(value)
	
	def set(self, symbol: str, value: RTValue):
		res = RTResult()
		if self.get(symbol).error != None:
			return res.fail(RTError(f"Symbol '{symbol}' does not exist."))
		if symbol in self.constants:
			return res.fail(RTError(f"Cannot change the value of constant '{symbol}'."))
		self.symbols[symbol] = value
		return res.success(Null())

## RUNTIME RESULT
class RTResult:
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.value: RTValue|None = None
		self.error: RTError|None = None
		self.return_value: RTValue = Null()
		self.loop_should_continue: bool = False
		self.loop_should_break: bool = False
	
	def register(self, res: Self) -> RTValue:
		self.error = res.error
		self.return_value = res.return_value
		self.loop_should_continue = res.loop_should_continue
		self.loop_should_break = res.loop_should_break
		return res.value
	
	def success(self, value: RTValue) -> Self:
		self.reset()
		self.value = value
		return self
	
	def success_return(self, value: RTValue) -> Self:
		self.reset()
		self.return_value = value
		return self
	
	def success_continue(self) -> Self:
		self.reset()
		self.loop_should_continue = True
		return self
	
	def success_break(self) -> Self:
		self.reset()
		self.loop_should_break = True
		return self
	
	def should_return(self) -> bool:
		if self.error != None: return True
		if not isinstance(self.return_value, Null): return True
		if self.loop_should_break: return True
		if self.loop_should_continue: return True
		return False

	def fail(self, error: RTError) -> Self:
		self.reset()
		self.error = error
		return self

## INTERPRETER
class Interpreter:
	def __init__(self):
		self.dataTypeConfig = {
			"integer": "Integer",
			"real": "Real",
			"boolean": "Boolean",
			"string": "String",
			"char": "Char",
			"object": "Object",
			"file": "FileHandler",
			"function": "Function",

			"array[integer]": "Array",
			"array[real]": "Array",
			"array[boolean]": "Array",
			"array[string]": "Array",
			"array[char]": "Array"
		}

	def visit(self, node: Program, env: Environment, output) -> RTResult:
		methodName: str = f"visit{type(node).__name__}"
		method = getattr(self, methodName, self.noVisitMethod)
		return method(node, env, output)
	
	def noVisitMethod(self, node: Program, env: Environment, output) -> RTResult:
		raise Exception(f"No visit{type(node).__name__} method defined.")
	
	def visitProgram(self, node: Program, env: Environment, output) -> RTResult:
		res = RTResult()
		lastResult = Null()
		for line in node.body:
			lastResult = res.register(self.visit(line, env, output))
			if res.should_return(): return res
		return res.success(lastResult)
	
	def visitIntLiteral(self, node: IntLiteral, env: Environment, output) -> RTResult:
		return RTResult().success(Integer(node.value))
	
	def visitRealLiteral(self, node: RealLiteral, env: Environment, output) -> RTResult:
		return RTResult().success(Real(node.value))
	
	def visitStringLiteral(self, node: StringLiteral, env: Environment, output) -> RTResult:
		return RTResult().success(String(node.value))
	
	def visitCharLiteral(self, node: CharLiteral, env: Environment, output) -> RTResult:
		return RTResult().success(Char(node.value))

	def visitArrayLiteral(self, node: ArrayLiteral, env: Environment, output) -> RTResult:
		res = RTResult()
		array: list[RTValue] = []
		for item in node.values:
			value = res.register(self.visit(item, env, output))
			if res.should_return(): return res
			array.append(value)
		
		return res.success(Array(array))

	def visitObjectLiteral(self, node: ObjectLiteral, env: Environment, output) -> RTResult:
		res = RTResult()
		properties: dict[str, RTValue] = dict()
		obj = Object(node.symbol, properties, env)
		for prop in node.properties:
			if res.should_return(): return res

			value = res.register(self.visit(prop.value, env if prop.dataType != "method" else obj, output))
			if res.should_return(): return res
			match prop.dataType:
				case "integer":
					if not isinstance(value, Integer): return res.fail(RTError("Value must be an integer."))
				case "real":
					if not isinstance(value, (Integer, Real)): return res.fail(RTError("Value must be a real."))
					value = Real(float(value.value))
				case "boolean":
					if not isinstance(value, (Boolean)): return res.fail(RTError("Value must be a boolean."))
				case "string":
					if not isinstance(value, (String)): return res.fail(RTError("Value must be a string."))
				case "char":
					if not isinstance(value, (Char)): return res.fail(RTError("Value must be a character."))
				case "file":
					if not isinstance(value, FileHandler): return res.fail(RTError("Value must be a file."))
				case "array[integer]":
					if not isinstance(value, Array):
						return res.fail(RTError("Value must be an integer array."))
					if value.arrayType != "integer":
						return res.fail(RTError("Value must be an integer array."))
				case "array[real]":
					if not isinstance(value, Array):
						return res.fail(RTError("Value must be a real array."))
					if value.arrayType != "real":
						return res.fail(RTError("Value must be a real array."))
				case "array[string]":
					if not isinstance(value, Array):
						return res.fail(RTError("Value must be a string array."))
					if value.arrayType != "string":
						return res.fail(RTError("Value must be a string array."))
				case "array[boolean]":
					if not isinstance(value, Array):
						return res.fail(RTError("Value must be a boolean array."))
					if value.arrayType != "boolean":
						return res.fail(RTError("Value must be a boolean array."))
				case "array[char]":
					if not isinstance(value, Array):
						return res.fail(RTError("Value must be a character array."))
					if value.arrayType != "char":
						return res.fail(RTError("Value must be a character array."))

			properties[prop.key] = value
		obj.properties = properties
		obj.scope.symbols = properties
		return env.declare(node.symbol, True, obj)

	def visitIdentifier(self, node: Identifier, env: Environment, output) -> RTResult:
		return env.get(node.value)
	
	def visitTernaryOperation(self, node: TernaryOperation, env: Environment, output) -> RTResult:
		res = RTResult()
		condition = res.register(self.visit(node.condition, env, output))
		if res.should_return(): return res
		value = res.register(self.visit(node.truthyValue if condition else node.falsyValue, env, output))
		if res.should_return(): return res
		return res.success(value)

	def visitBinaryOperation(self, node: BinaryOperation, env: Environment, output) -> RTResult:
		res = RTResult()
		left = res.register(self.visit(node.left, env, output))
		if res.should_return(): return res
		right = res.register(self.visit(node.right, env, output))
		if res.should_return(): return res
		match node.operator.value:
			case "+": result, error = left + right
			case "-": result, error = left - right
			case "*": result, error = left * right
			case "/": result, error = left / right
			case "%": result, error = left % right
			case "^": result, error = left ** right
			case "<": result, error = left < right
			case "<=": result, error = left <= right
			case "==": result, error = left == right
			case "!=": result, error = left != right
			case ">": result, error = left > right
			case ">=": result, error = left >= right
			case "<<": result, error = left << right
			case ">>": result, error = left >> right
			case "and": result, error = left & right
			case "or": result, error = left | right

		if error: return res.fail(error)
		return res.success(result)

	def visitUnaryOperation(self, node: UnaryOperation, env: Environment, output) -> RTResult:
		res = RTResult()
		expr = res.register(self.visit(node.expr, env, output))
		if res.should_return(): return res
		match node.operator.value:
			case "-": expr, error = Integer(0) - expr
			case "+": expr, error = expr, None
			case "not": expr, error = ~expr
		
		if error: return res.fail(error)
		return res.success(expr)

	def visitVarDeclaration(self, node: VarDeclaration, env: Environment, output) -> RTResult:
		res = RTResult()
		if node.value != None:
			value = res.register(self.visit(node.value, env, output))
			if res.should_return(): return res

			if self.dataTypeConfig.get(node.dataType, None) != type(value).__name__:
				if not (type(value).__name__ == "Integer" and node.dataType == "real"):
					return res.fail(RTError(f"Cannot convert type<{node.dataType}> to type<{type(value).__name__.lower()}>."))
				else: value = Real(float(value.value))
			
			if node.dataType.startswith("array["):
				value: Array
				if value.arrayType.lower() != node.dataType[6:-1]:
					return res.fail(RTError(f"Expected {'an' if node.dataType[6] in 'aeiou' else 'a'} {node.dataType[6:-1]} array, found {'an' if value.arrayType.lower()[0] in 'aeiou' else 'a'} {value.arrayType.lower()} array instead."))
			
			if isinstance(value, Function):
				value.name = node.symbol
		else:
			match node.dataType:
				case "integer": value = Integer(0)
				case "real": value = Real(0.0)
				case "boolean": value = Boolean(False)
				case "string": value = String("")
				case "char": value = Char()
				case "file": value = FileHandler("")
				case "function": value = Function("<null>", [], [], None, None)

				case "array[integer]": value = Array([], "integer")
				case "array[real]": value = Array([], "real")
				case "array[boolean]": value = Array([], "boolean")
				case "array[string]": value = Array([], "string")
				case "array[char]": value = Array([], "char")
		if res.should_return(): return res
		return env.declare(node.symbol, node.isConst, value)

	def visitAssignment(self, node: Assignment, env: Environment, output) -> RTResult:
		res = RTResult()
		value = res.register(self.visit(node.value, env, output))
		if res.should_return(): return res

		data = res.register(env.get(node.symbol))
		if res.should_return(): return res

		# Type check
		match type(data).__name__:
			case "Integer":
				if not isinstance(value, Integer): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <integer>.")
			case "Real":
				if not isinstance(value, (Integer, Real)):
					return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <real>.")
				value = Real(float(value.value))
			case "Boolean":
				if not isinstance (value, Boolean): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <boolean>.")
			case "String":
				if not isinstance (value, String): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <string>.")
			case "Char":
				if not isinstance (value, Char): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <char>.")
			case "Array":
				data: Array
				if not isinstance (value, Array): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <array>.")
				if value.arrayType.lower() != data.arrayType.lower():
					return res.fail(RTError(f"Expected {'an' if data.arrayType.lower() in 'aeiou' else 'a'} {data.arrayType.lower()} array, found {'an' if value.arrayType.lower()[0] in 'aeiou' else 'a'} {value.arrayType.lower()} array instead."))
			case "FileHandler":
				if not isinstance (value, FileHandler): return res.fail(f"Cannot assign type <{type(value).__name__.lower()}> to type <file>.")
		
		# Set value
		if isinstance(value, Function):
			value.name = node.symbol

		result = value
		error = None
		match node.operation:
			case "+=": result, error = data + value
			case "-=": result, error = data - value
			case "*=": result, error = data * value
			case "/=": result, error = data / value
			case "%=": result, error = data % value
			case "^=": result, error = data ** value
		
		if error: return res.fail(error)
		return env.set(node.symbol, result)

	def visitForExpression(self, node: ForExpression, env: Environment, output) -> RTResult:
		res = RTResult()

		startValue = res.register(self.visit(node.startValue, env, output))
		if res.should_return(): return res
		if not isinstance(startValue, Integer) or (isinstance(startValue, Real) and startValue.value != 0.0):
			return res.fail(RTError("Start value must be an integer."))
		
		endValue = res.register(self.visit(node.endValue, env, output))
		if res.should_return(): return res
		if not isinstance(endValue, Integer) or (isinstance(endValue, Real) and endValue.value != 0.0):
			return res.fail(RTError("End value must be an integer."))
		
		if node.stepValue != None:
			stepValue = res.register(self.visit(node.stepValue, env, output))
			if res.should_return(): return res
			if not isinstance(stepValue, Integer) or (isinstance(stepValue, Real) and stepValue.value != 0.0):
				return res.fail(RTError("Step value must be an integer."))
		else:
			stepValue = Integer(1)
		if stepValue.value == 0:
			return res.fail(RTError("Step value cannot be 0."))
		
		# Setup counter
		if env.get(node.symbol).error: env.declare(node.symbol, False, startValue)
		else: env.set(node.symbol, startValue)
		i = startValue.value
		if stepValue.value > 0: condition = lambda : i <= endValue.value
		else: condition = lambda : i >= endValue.value

		while condition():
			env.set(node.symbol, Integer(i))
			i += stepValue.value
			res.register(self.visit(Program(node.body), env, output))

			if res.should_return() and res.loop_should_break == False and res.loop_should_continue == False:
				return res
			if res.loop_should_continue: continue
			if res.loop_should_break: break

		return res.success(Null())

	def visitWhileExpression(self, node: WhileExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		
		while True:
			condition = res.register(self.visit(node.condition, env, output))
			if res.should_return(): return res
			if not condition: break
			# Execute body
			res.register(self.visit(Program(node.body), env, output))

			if res.should_return() and res.loop_should_break == False and res.loop_should_continue == False:
				return res
			if res.loop_should_continue: continue
			if res.loop_should_break: break
		return res.success(Null())
	
	def visitRepeatExpression(self, node: RepeatExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		lastResult = Null()
		while True:
			# Execute body
			lastResult = res.register(self.visit(Program(node.body), env, output))
			if res.should_return(): return res
			# Check condition
			condition = res.register(self.visit(node.condition, env, output))
			if res.should_return(): return res
			if condition: break
		return res.success(lastResult)

	def visitIfExpression(self, node: IfExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		for condition, body in node.cases:
			conditionValue = res.register(self.visit(condition, env, output))
			if res.should_return(): return res
			if bool(conditionValue):
				bodyValue = res.register(self.visit(Program(body), env, output))
				if res.should_return(): return res
				return res.success(Null())
		# Else case
		if node.elseCase != None:
			bodyValue = res.register(self.visit(Program(node.elseCase), env, output))
			if res.should_return(): return res
			return res.success(Null())
		return res.success(Null())

	def visitFunctionDeclaration(self, node: FunctionDeclaration, env: Environment, output) -> RTResult:
		res = RTResult()
		func = Function(node.functionName, node.args, node.body, env, node.should_auto_return, len(node.args))
		declaration = env.declare(node.functionName, True, func)
		if declaration.error: return declaration
		return res.success(func)

	def visitMethodDeclaration(self, node: MethodDeclaration, obj: Object, output) -> RTResult:
		res = RTResult()
		method = Method(node.methodName, node.args, node.body, obj, len(node.args))
		declaration = obj.scope.declare(node.methodName, True, method)
		if declaration.error: return declaration
		return res.success(method)

	def visitCallExpression(self, node: CallExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		func: Function|NativeFunction = res.register(self.visit(node.identifier, env, output))

		if res.should_return(): return res
		return func(node.args, output)

	def visitTryExpression(self, node: TryExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		tryValue = res.register(self.visit(Program(node.tryBody), env, output))
		if not res.error:
			if node.otherwiseBody == []: return res.success(tryValue)
			else:
				otherwiseValue = res.register(self.visit(Program(node.otherwiseBody), env, output))
				if res.should_return(): return res
				return res.success(otherwiseValue)
		catchValue = res.register(self.visit(Program(node.catchBody), env, output))
		if res.should_return(): return res
		return res.success(catchValue)

	def visitSwitchStatement(self, node: SwitchStatement, env: Environment, output) -> RTResult:
		res = RTResult()
		parameter: RTValue = res.register(self.visit(node.parameter, env, output))
		if res.should_return(): return res
		executed: bool = False

		for case in node.cases:
			test_value: RTValue = res.register(self.visit(case[0], env, output))

			if test_value.value == parameter.value: # Comparison is a Boolean
				res.register(self.visit(case[1], env, output))
				if res.should_return(): return res
				executed = True
		
		if node.otherwise_case and not executed:
			res.register(self.visit(node.otherwise_case, env, output))
			if res.should_return(): return res
		
		return res.success(Null())

	def visitArrayAccessExpression(self, node: ArrayAccessExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		array = res.register(self.visit(node.array, env, output))
		if res.should_return(): return res
		if not isinstance(array, (Array, String)):
			return res.fail(f"Cannot access <{type(array).__name__.lower()}>.")
		
		index = res.register(self.visit(node.index, env, output))
		if res.should_return(): return res
		if not isinstance(index, Integer):
			if isinstance(index, Real) and index.value == int(index.value):
				index = Integer(int(index.value))
			else:
				return res.fail(RTError("Array index must evaluate to an integer."))
		
		length: int = len(array.value)
		if index.value not in list(range(length)):
			return res.fail(RTError("Array index out of range!"))
		
		if isinstance(array, Array): return res.success(array.value[index.value])
		return res.success(String(array.value[index.value]))

	def visitArraySetExpression(self, node: ArraySetExpression, env: Environment, output) -> RTResult:
		res = RTResult()
		array = res.register(self.visit(node.array, env, output))
		if res.should_return(): return res
		if not isinstance(array, Array):
			return res.fail(RTError(f"Cannot change an item of a value of type <{type(array).__name__.lower()}>."))

		index = res.register(self.visit(node.index, env, output))
		if res.should_return(): return res
		if not isinstance(index, Integer):
			if isinstance(index, Real) and index.value == int(index.value):
				index = Integer(int(index.value))
			else:
				return res.fail(RTError("Array index must evaluate to an integer."))
		length: int = len(array.value)
		if index.value not in list(range(length)):
			return res.fail(RTError("Array index out of range!"))

		value = res.register(self.visit(node.value, env, output))
		if res.should_return(): return res
		if type(value).__name__.lower() != array.arrayType:
			if type(value).__name__.lower() == "integer" and array.arrayType == "real":
				pass
			else:
				return res.fail(array.illegalOperation(value)[1]) # Extracts the error
		
		match node.operation:
			case "=": array.value[index.value] = value
			case "+=": array.value[index.value] = (array.value[index.value] + value)[0]
			case "-=": array.value[index.value] = (array.value[index.value] - value)[0]
			case "*=": array.value[index.value] = (array.value[index.value] * value)[0]
			case "/=":
				if value == 0: return res.fail(RTError("Division by 0!"))
				array.value[index.value] = (array.value[index.value] / value)[0]
			case "%=": array.value[index.value] = (array.value[index.value] % value)[0]
			case "**=": array.value[index.value] = (array.value[index.value] ** value)[0]
		
		return res.success(Null())

	def visitReturnNode(self, node: ReturnNode, env: Environment, output):
		res = RTResult()

		if node.node_to_return != None:
			value = res.register(self.visit(node.node_to_return, env, output))
			if res.should_return(): return res
		else:
			value = Null()
		
		return res.success_return(value)

	def visitContinueNode(self, node: ContinueNode, env: Environment, output):
		return RTResult().success_continue()
	
	def visitBreakNode(self, node: BreakNode, env: Environment, output):
		return RTResult().success_break()

	def visitMemberExpression(self, node: MemberExpression, env: Environment, output):
		res = RTResult()
		obj = res.register(self.visit(node.object, env, output))
		if res.error: return res

		if not isinstance(obj, Object):
			return res.fail(RTError("Member expressions can only be applied to objects."))
		
		try:
			value = obj.properties[node.property.value]
		except:
			return res.fail(RTError(f"Cannot get property '{node.property.value}' of object {obj.symbol}."))
		return res.success(value)

## RUN
def run(fn: str, ftxt: str, env: Environment, output):
	project.terminalEdit.setReadOnly(True)

	lexer = Lexer(fn, ftxt)
	tokens, error = lexer.lex()
	if error: return None, error

	parser = Parser(fn, tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error

	interpreter = Interpreter()
	result = interpreter.visit(ast.node, env, output)
	if result.error: return None, result.error
	return result.value, None

## Setup
globalTable = Environment()

def resetSymbols():
	table = Environment()

		# Built-in symbols
	table.declare("true", True, Boolean(True))
	table.declare("false", True, Boolean(False))
	table.declare("endl", True, String("\n"))
	table.declare("null", True, Null())

		# System functions
	table.declare("print", True, NativeFunction("Print", [], table))
	table.declare("read", True, NativeFunction("Read", [], table, 0))
	table.declare("error", True, NativeFunction("Error", [("message", "string")], table, 1))

		# Conversion functions
	table.declare("to_string", True, NativeFunction("ToString", [("value", "null")], table, 1))
	table.declare("to_integer", True, NativeFunction("ToInteger", [("value", "null")], table, 1))
	table.declare("to_real", True, NativeFunction("ToReal", [("value", "null")], table, 1))
	table.declare("to_boolean", True, NativeFunction("ToBoolean", [("value", "null")], table, 1))
	table.declare("to_char", True, NativeFunction("ToChar", [("value", "null")], table, 1))

		# Type-checking functions
	table.declare("is_string", True, NativeFunction("IsString", [("value", "null")], table, 1))
	table.declare("is_integer", True, NativeFunction("IsInteger", [("value", "null")], table, 1))
	table.declare("is_real", True, NativeFunction("IsReal", [("value", "null")], table, 1))
	table.declare("is_boolean", True, NativeFunction("IsBoolean", [("value", "null")], table, 1))
	table.declare("is_char", True, NativeFunction("IsChar", [("value", "null")], table, 1))

		# File handling functions
	table.declare("open_file", True, NativeFunction("Openfile", [("fn", "string"), ("mode", "char")], table, 2))
	table.declare("close_file", True, NativeFunction("Closefile", [("file", "file")], table, 1))
	table.declare("read_file", True, NativeFunction("Readfile", [("file", "file")], table, 1))
	table.declare("write_file", True, NativeFunction("Writefile", [("file", "file"), ("content", "string")], table, 2))
	table.declare("is_eof", True, NativeFunction("Iseof", [("file", "file")], table, 1))

		# Other
	table.declare("format", True, NativeFunction("Format", [], table))
	table.declare("pow", True, NativeFunction("Pow", [], table))
	table.declare("type", True, NativeFunction("Type", [("value", "null")], table, 1))
	table.declare("length", True, NativeFunction("Length", [("value", "null")], table, 1))
	table.declare("reverse", True, NativeFunction("Reverse", [("value", "null")], table, 1))
	table.declare("sorted", True, NativeFunction("Sorted", [("arr", "array")], table, 1))
	table.declare("sum", True, NativeFunction("Sum", [("arr", "array")], table, 1))

	table.declare("max", True, NativeFunction("Max", [("x", "null"), ("y", "null")], table, 2))
	table.declare("min", True, NativeFunction("Min", [("x", "null"), ("y", "null")], table, 2))
	table.declare("rand", True, NativeFunction("Rand", [], table, 0))
	table.declare("randint", True, NativeFunction("Randint", [("start", "integer"), ("end", "integer")], table, 2))

	return table

globalTable.symbols = resetSymbols().symbols

## SYNTAX HIGHLIGHTING
def format(color, style: str = ""):
	_color = QColor()
	_color.setNamedColor(color)

	_format = QTextCharFormat()
	_format.setForeground(_color)

	# Set font weight
	if 'bold' in style:
		_format.setFontWeight(300)
	if 'italic' in style:
		_format.setFontItalic(True)
	return _format

STYLES = {
	"keyword": format("#53AFFF", "bold"),
	"operator": format("#F668C9", "bold"),
	"brace": format("#9DD728"),
	"string": format("#E5C44F"),
	"comment": format("#666D6F", "italic"),
	"number": format("#8080DD")
}

class Highlighter(QSyntaxHighlighter):
	KEYWORDS = Lexer("", "").KEYWORDS
	OPERATORS = ["+", "-", "\*", "/", "^", "%", "==", "!=", "<=", ">=", "<", ">", ":", ",", "?"]
	BRACES = ["{", "}", "(", ")", "[", "]"]

	def __init__(self, parent: QTextDocument):
		super().__init__(parent)
		rules: list[tuple[str, int, QTextCharFormat]] = []

		# Keyword, operator & brace rules
		rules += [(r'\b%s\b' % kw, 0, STYLES['keyword']) for kw in self.KEYWORDS]
		rules += [(r'%s' % op, 0, STYLES['operator']) for op in self.OPERATORS]
		rules += [(r'%s' % br, 0, STYLES['brace']) for br in self.BRACES]

		self.rules = [(QRegularExpression(pattern), index, fmt) for (pattern, index, fmt) in rules]

	def highlightBlock(self, text):
		for expression, index, fmt in self.rules:
			index = expression.match()

## QMAINWINDOWS
user: str = ""
projectName: str = ""
version: str = "1.1"

class Login(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/login.ui", self)
		self.inputPass: str = ""
		self.setWindowTitle("Login")
		self.toRegisterButton.clicked.connect(self.openRegisterPage)
		self.loginButton.clicked.connect(self.pressedLogin)
	
	def openRegisterPage(self):
		self.close()
		register.show()
	
	def pressedLogin(self):
		global user
		# Update username and password
		user = self.usernameEdit.text()
		inputPass = self.passwordEdit.text()

		if accounts.get(user, None) == inputPass:
			# Open home page
			self.errorMessage.setText("")
			if not self.rememberMe.isChecked():
				self.usernameEdit.setText("")
				self.passwordEdit.setText("")
				register.usernameEdit.setText("")
				register.passwordEdit.setText("")
			self.close()

			home.welcomeLabel.setText(f"⌂ Welcome, {user}!")
			home.displayProjects(user)
			home.show()
		elif user in accounts:
			self.errorMessage.setText("Incorrect password!")
		else:
			self.errorMessage.setText(f"Cannot find username '{user}'.")

	def keyPressEvent(self, a0):
		if a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
			self.pressedLogin()
			return

		return super().keyPressEvent(a0)

class Register(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/signup.ui", self)
		self.setWindowTitle("Register")
		self.toLoginButton.clicked.connect(self.openLoginPage)
		self.registerButton.clicked.connect(self.pressedRegister)
	
	def openLoginPage(self):
		self.close()
		login.show()

	def pressedRegister(self):
		global user
		user = self.usernameEdit.text()
		password = self.passwordEdit.text()
		confirmPass = self.confirmPassEdit.text()

		if user == "":
			self.errorMessage.setText("Username field cannot be blank.")
		elif password == "":
			self.errorMessage.setText("Password field cannot be blank.")
		elif confirmPass == "":
			self.errorMessage.setText("Confirm passowrd field cannot be blank.")
		elif user in accounts:
			self.errorMessage.setText(f"User {user} already exists.")
		elif password != confirmPass:
			self.errorMessage.setText("Cannot confirm password.")
		else:
			accounts[user] = password
			jsonStr = json.dumps(accounts)

			with open("accounts.json", "w") as file:
				file.write(jsonStr)

			self.usernameEdit.setText("")
			self.passwordEdit.setText("")
			login.usernameEdit.setText("")
			login.passwordEdit.setText("")
			self.close()

			self.errorMessage.setText("")
			home.welcomeLabel.setText(f"⌂ Welcome, {user}!")
			os.mkdir(f"Projects/{user}")
			with open(f"Projects/{user}/recentProjects.config", "w") as _: ...
			home.displayProjects(user)
			home.show()

	def keyPressEvent(self, a0):
		if a0.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
			self.pressedRegister()
			return

		return super().keyPressEvent(a0)

class SignOut(QDialog):
	def __init__(self):
		super().__init__()
		self.setFixedSize(400, 180)
		self.setFont(QFont("JetBrains Mono", 9))
		self.setWindowTitle("Sign Out Confirmation")
		self.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(54, 54, 90, 255), stop:1 rgba(0, 0, 255, 255));")

		self.label = QLabel(self)
		self.label.setText("Are you sure that you want to sign out?")
		self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.label.setFont(QFont("JetBrains Mono", 12, 800))
		self.label.setGeometry(10, 50, 380, 20)

		self.yes = QPushButton(self)
		self.yes.setText("Yes")
		self.yes.setGeometry(40, 95, 100, 30)
		self.yes.setStyleSheet('QPushButton {\nbackground-color: rgb(0, 85, 0); color: white; font: 10pt "JetBrains Mono NL"; border: 2px solid white;\n}\nQPushButton:hover {\nbackground-color: rgb(0, 255, 0); color: black; border: 2px solid black;\n}')
		self.yes.setCursor(Qt.CursorShape.PointingHandCursor)
		self.yes.clicked.connect(self.accept)

		self.no = QPushButton(self)
		self.no.setText("No")
		self.no.setGeometry(260, 95, 100, 30)
		self.no.setStyleSheet('QPushButton {\nbackground-color: rgb(85, 0, 0); color: white; font: 10pt "JetBrains Mono NL"; border: 2px solid white;\n}\nQPushButton:hover {\nbackground-color: rgb(255, 0, 0); color: black; border: 2px solid black;\n}')
		self.no.setCursor(Qt.CursorShape.PointingHandCursor)
		self.no.clicked.connect(self.reject)

class Home(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/main.ui", self)
		self.welcomeLabel.setText(f"Welcome, {user}!")
		self.setWindowTitle("Home")
		self.projectsButton.clicked.connect(self.openProjects)
		self.settingsButton.clicked.connect(self.openSettings)
		self.newProjectButton.clicked.connect(self.openNewProject)
		self.signOut.clicked.connect(self.openLogin)

	def displayProjects(self, username: str):
		with open(f"Projects/{username}/recentProjects.config") as file:
			projects = [p.strip() for p in file.readlines()]
			while "" in projects:
				projects.remove("")
			projectLabels: list[QPushButton] = [self.project1, self.project2, self.project3, self.project4]

			for button in projectLabels:
				button.setText("")
				button.setEnabled(False)
				try:
					button.disconnect()
				except TypeError:
					pass

			for i, name in enumerate(projects):
				projectLabels[i].setText(name)
				projectLabels[i].setEnabled(True)
				projectLabels[i].setCursor(Qt.CursorShape.PointingHandCursor)
				projectLabels[i].clicked.connect(partial(self.openProject, name))

	def openProject(self, fn: str):
		# TODO: determine file name
		with open(f"Projects/{user}/{fn}", "r") as file:
			ftxt = ""
			for line in file:
				ftxt += line
			else:
				# Done with the file
				project.codeEdit.setFontPointSize(settings.editorFontSize.value())
			
				project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
				project.codeEdit.setText(ftxt)
				project.projectName.setText(fn)
		
		path: str = f"Projects/{user}"
		with open(f"{path}/recentProjects.config", "r") as recentProjects:
			projects = [p.strip("\n") for p in recentProjects.readlines()]
			while "" in projects:
				projects.remove("")
		projects.remove(fn)

		with open(f"{path}/recentProjects.config", "w") as recentProjects:
			recentProjects.write(fn + "\n")
			recentProjects.write("\n".join(projects[0:3] if len(projects) >= 4 else projects))

		self.close()

		project.terminalEdit.setText("")
		project.codeEdit.setFontPointSize(settings.editorFontSize.value())
	
		project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
		project.setWindowTitle(f"Project '{fn}'")
		project.icon.setPixmap(QPixmap("Images/pyscript.png" if fn.endswith(".ps") else "Images/txt.png"))
		project.show()

	def openProjects(self):
		self.close()
		projects.show()
		projects.listProjects()
			
	def openNewProject(self):
		if pName.exec() == 1:
			pName.close()
			self.close()
			project.codeEdit.setText("")
			project.terminalEdit.setText("")
			project.codeEdit.setFontPointSize(settings.editorFontSize.value())
		
			project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
			project.show()

	def openSettings(self):
		self.close()
		settings.show()

	def openLogin(self):
		if signOut.exec() == 1:
			self.close()
			login.show()

class Projects(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/projects.ui", self)
		self.setWindowTitle("Projects")
		self.homeButton.clicked.connect(self.openHome)
		self.settingsButton.clicked.connect(self.openSettings)
		self.newProjectButton.clicked.connect(self.openProject)
		self.signOut.clicked.connect(self.openLogin)

	def listProjects(self):
		global user
		path = f"Projects/{user}"
		allProjects = os.listdir(path)
		listedProjects: list[str] = [p for p in allProjects if p.endswith((".ps", ".txt"))]

		projectButtons: list[QPushButton] = [getattr(self, f"project_{i}") for i in range(14)]

		for button in projectButtons:
			button.setEnabled(False)
			button.setText("")
			button.setIcon(QIcon())
			try:
				button.disconnect()
			except TypeError:
				pass

		shownProjects = listedProjects[-14:]
		for i, file in enumerate(shownProjects):
			projectButtons[i].setIcon(QIcon("Images/pyscript.png" if file.endswith(".ps") else "Images/txt.png"))
			projectButtons[i].setText(file)
			projectButtons[i].setEnabled(True)
		
			projectButtons[i].clicked.connect(partial(self.openFile, fn=file))
	
	def openHome(self):
		self.close()
		home.show()
		home.displayProjects(user)
	
	def openProject(self):
		if pName.exec() == 1:
			pName.close()
			self.close()
			project.codeEdit.setText("")
			project.terminalEdit.setText("")
			project.codeEdit.setFontPointSize(settings.editorFontSize.value())
		
			project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
			project.show()
	
	def openSettings(self):
		self.close()
		settings.show()

	def openLogin(self):
		if signOut.exec() == 1:
			self.close()
			login.show()

	def openFile(self, fn: str):
		# TODO: determine file name
		path: str = f"Projects/{user}"

		with open(f"{path}/{fn}", "r") as file:
			ftxt = ""
			for line in file:
				ftxt += line
			else:
				# Done with the file
				project.codeEdit.setFontPointSize(settings.editorFontSize.value())
				project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
				project.codeEdit.setText(ftxt)
				project.projectName.setText(fn)
		
		with open(f"{path}/recentProjects.config", "r") as recentProjects:
			projects = [p.strip("\n") for p in recentProjects.readlines()]
			while "" in projects:
				projects.remove("")
		if fn in projects: projects.remove(fn)

		with open(f"{path}/recentProjects.config", "w") as recentProjects:
			recentProjects.write(fn + "\n")
			recentProjects.write("\n".join(projects[0:3] if len(projects) >= 4 else projects))

		self.close()

		project.terminalEdit.setText("")
		project.codeEdit.setFontPointSize(settings.editorFontSize.value())
	
		project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
		project.setWindowTitle(f"Project '{fn}'")
		project.icon.setPixmap(QPixmap("Images/pyscript.png" if fn.endswith(".ps") else "Images/txt.png"))
		project.show()

class Settings(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/settings.ui", self)
		self.setWindowTitle("Settings")
		self.homeButton.clicked.connect(self.openHome)
		self.projectsButton.clicked.connect(self.openProjects)
		self.signOut.clicked.connect(self.openLogin)
	
	def openHome(self):
		self.close()
		home.show()
		home.displayProjects(user)

	def openProjects(self):
		self.close()
		projects.show()
		projects.listProjects()

	def openLogin(self):
		if signOut.exec() == 1:
			self.close()
			login.show()

class GetProjectName(QDialog):
	def __init__(self):
		super().__init__()
		self.setFixedSize(400, 180)
		self.setFont(QFont("JetBrains Mono", 9))
		self.setWindowTitle("Let's create something...")
		self.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(54, 54, 90, 255), stop:1 rgba(0, 0, 255, 255));")

		self.projectName = QLineEdit(self)
		self.projectName.setPlaceholderText("Enter the project name...")
		self.projectName.setGeometry(20, 20, 360, 30)
		self.projectName.setStyleSheet("background-color: rgb(38, 38, 67);")
		self.projectName.setText("")

		self.errorMessage = QLabel(self)
		self.errorMessage.setStyleSheet("color: red")
		self.errorMessage.setText("")
		self.errorMessage.setGeometry(20, 120, 360, 30)

		self.confirmButton = QPushButton(self)
		self.confirmButton.setText("Let's go!")
		self.confirmButton.setGeometry(100, 70, 200, 30)
		self.confirmButton.setStyleSheet('QPushButton { background-color: rgb(0, 85, 255);	color: rgb(255, 255, 255);	font: 10pt "JetBrains Mono NL";	border: 2px solid white; } QPushButton:hover {\nbackground-color: rgb(85, 170, 255);color: rgb(0, 0, 0);border: 2px solid black; }')
		self.confirmButton.setCursor(Qt.CursorShape.PointingHandCursor)
		self.confirmButton.clicked.connect(self.confirm)
	
	def confirm(self):
		name: str = self.projectName.text()
		if not name.endswith((".ps", ".txt")):
			self.errorMessage.setText("Project must end with '.ps' or '.txt'.")
		else:
			path: str = f"Projects/{user}"
			allProjects = os.listdir(path)
			listedProjects: list[str] = [p for p in allProjects if p.endswith((".ps", ".txt"))]
			if name in listedProjects:
				self.errorMessage.setText(f"Project '{name}' already exists.")
			else:
				with open(f"{path}/recentProjects.config", "r") as recentProjects:
					projects = [p.strip("\n") for p in recentProjects.readlines()]
					while "" in projects:
						projects.remove("")
				with open(f"{path}/recentProjects.config", "w") as recentProjects:
					recentProjects.write(f"{name}\n")
					recentProjects.write("\n".join(projects[0:3] if len(projects) >= 4 else projects))

				project.projectName.setText(name)
				project.codeEdit.setText("")
				project.terminalEdit.setText("")
				project.codeEdit.setFontPointSize(settings.editorFontSize.value())
			
				project.codeEdit.setTabStopDistance(settings.tabSize.value() * 26.5 / 4)
				project.setWindowTitle(f"Project '{name}'")
				project.icon.setPixmap(QPixmap("Images/pyscript.png" if name.endswith(".ps") else "Images/txt.png"))
				file = open(f"{path}/{name}", "w")
				file.close()
				self.accept()

class Project(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi("GUI/code.ui", self)
		self.terminalEdit.setReadOnly(True)
		self.codeEdit: QTextEdit
		self.terminalEdit: QTextEdit
		self.codeEdit.setPlaceholderText("Enter some code...")
		self.loop = QEventLoop()

		self.runButton.clicked.connect(self.runCode)
		self.clearButton.clicked.connect(self.terminalEdit.clear)
		self.homeButton.clicked.connect(self.openHome)
		self.terminalEdit.installEventFilter(self)
		self.codeEdit.installEventFilter(self)

	def openHome(self):
		with open(f"Projects/{user}/{self.projectName.text()}", "w") as file:
			file.write(self.codeEdit.document().toPlainText())
		self.close()
		home.show()
		home.displayProjects(user)

	def runCode(self):
		# Don't run if it is a text file
		if self.projectName.text().endswith(".txt"):
			return

		# Reset
		self.terminalEdit.document().clear()
		self.terminalEdit.append(f"PyScript v{version}\n")
		self.terminalEdit.setReadOnly(True)
		globalTable.symbols = resetSymbols().symbols
		self.loop.exit()

		# Run code
		_, error = run(
			"<code>", # File name
			self.codeEdit.document().toPlainText(), # File text
			globalTable, # Environment
			self
		)
		if error:
			self.terminalEdit.append(repr(error) + "\n--- CODE EXITED WITH ERRORS ---")
		else:
			self.terminalEdit.append("\n--- CODE RAN SUCCESSFULLY ---")
		self.terminalEdit.setReadOnly(True)

	def eventFilter(self, source: QObject|None, event: QEvent|None):

		terminal_cursor = self.terminalEdit.textCursor()
		if terminal_cursor.position() < input_pos:
			terminal_cursor.movePosition(QTextCursor.MoveOperation.End)
			self.terminalEdit.setTextCursor(terminal_cursor)
		
		if (source == self.terminalEdit and event.type() == QEvent.Type.KeyPress) and not self.terminalEdit.isReadOnly():
			# Process input
			if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
				self.terminalEdit.setReadOnly(True)
				self.loop.exit()
				return True
			
			# Safeguard from backspace
			elif event.key() == Qt.Key.Key_Backspace:
				if terminal_cursor.position() <= input_pos:
					return True
				else: return super().eventFilter(source, event)
			else: return super().eventFilter(source, event)
		elif source == self.codeEdit and event.type() == QEvent.Type.KeyPress:
			if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Slash:
				# Setup cursor
				code_cursor: QTextCursor = self.codeEdit.textCursor()

				# Toggle comment
				if not code_cursor.hasSelection():
					code_cursor.select(code_cursor.SelectionType.LineUnderCursor)
				
				selection_start: int = code_cursor.selectionStart()
				selection_end: int = code_cursor.selectionEnd()

				# Normalize selection to full lines
				code_cursor.setPosition(selection_start)
				code_cursor.movePosition(code_cursor.MoveOperation.StartOfBlock, code_cursor.MoveMode.KeepAnchor)
				selection_start = code_cursor.selectionStart()

				code_cursor.setPosition(selection_end)
				code_cursor.movePosition(code_cursor.MoveOperation.EndOfBlock, code_cursor.MoveMode.KeepAnchor)
				selection_end = code_cursor.selectionEnd()

				# Extract all lines in the selection range
				code_cursor.setPosition(selection_start)
				code_cursor.setPosition(selection_end, code_cursor.MoveMode.KeepAnchor)
				selected_text: str = code_cursor.selection().toPlainText()
				lines: list[str] = selected_text.splitlines()

				# Toggle comment for each line
				for i, line_text in enumerate(lines):
					line_text: str = line_text.strip()
					if line_text.startswith("// "):
						lines[i] = (line_text.lstrip().replace("// ", "", 1))
					elif line_text.startswith("//"):
						lines[i] = (line_text.lstrip().replace("//", "", 1))
					else:
						lines[i] = ("// " + line_text)
				
				# Update line
				updated_text = '\n'.join(lines)
				code_cursor.insertText(updated_text)
				return True
			else:
				return super().eventFilter(source, event)
		else:
			return super().eventFilter(source, event)

if __name__ == "__main__":
	app = QApplication(sys.argv)

	# Login & register pages
	login = Login()
	register = Register()

	# Dialogs
	pName = GetProjectName()
	signOut = SignOut()

	# Home pages
	home = Home()
	projects = Projects()
	settings = Settings()
	project = Project()
	login.show()
	sys.exit(app.exec())
