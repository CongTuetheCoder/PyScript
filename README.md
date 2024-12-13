# PyScript 1.0
A custom programming language made using Python.

## SYNTAX
### DATA TYPES - CAN BE ACCESSED BY 'type()' BUILT-IN FUNCTION
	type <integer>: 4, 31, -34
	type <real>: 0.4, -15.4
	type <string>: "Hello!"
	type <boolean>: true, false
	type <char>: 'a', 'e', '3'
	type <array>: [], ["a", "Hello"], [0.2, 10.5, 3.14159]
	type <object>: obj_name
	type <function>: function_name
	type <nativefunction>: print
	type <method>: obj_name.method
	type <typing>: type(1)
 	type <null>: print() // Calls print

### VARIABLE DECLARATION
	declare var_name: data_type = value
	const constant: data_type = value

### ARRAY DECLARATION IN VARIABLES
	declare array_name: array[data_type] = [value1, value2, value3, ...]
	const array_name: array[data_type] = [value1, value2, value3, ...]

### OPERATIONS
	num1 + num2
	num1 - num2
	num1 * num2
	num1 / num2
	num1 ^ num2
	num1 % num2
	-num

	variable = value
	variable += value
	variable -= value
	variable *= value
	variable /= value
	variable %= value
	variable ^= value

	num1 < num2
	num1 > num2
	num1 == num2
	num1 <= num2
	num1 >= num2
	num1 != num2

	num1 << num2
	num1 >> num2
	condition ? truthy_value : falsy_value

### OBJECT DECLARATION
	object obj_name {
		key: data type <- value
		method name(p1: data_type, p2: data_type, ...) {
			<body>
		}
	} // OOP is limited

### FUNCTION DECLARATION
	function func_name(p1: data_type, p2: data_type, ...) {
		<body>
		return <expression> // Returns the value of the expression.
	}

	inline func_name p1: data_type, p2: data_type, ... -> { <body> } // This automatically returns the last statement.

### LOOPS
	for i = startval -> endval (step stepval) {
		<body>
	}
	// Loops from start_val -> end_val

	while condition {
		<body>
	}
	// Executes until the condition is false, if it is false the first time then don't execute it

	repeat {
		<body>
	} until condition
	// Executes until the condition is false at least once

	continue // Skips the rest of the loop's body and executes the next iteration
	break // Exits out of the loop

### CONDITIONALS
	if <condition1> {
		// If <condition1> is true then execute
		<body1>
	} elif <condition2> {
		// If <condition1> is false and <condition2> is true then execute
		<body2>
	} else {
		// If all above conditions are false then execute
		<body3>
	}

	switch <parameter> {
		case <value1> {
			// If <parameter> == <value1> then execute <body1>
			<body1>
		} case <value2> {
			// Else if <parameter> == <value2> then execute <body2>
			<body2>
		} otherwise {
			// Otherwise case is optional, but it will execute if <parameter> != all other test values
			<body3>
		}
	}

### ERROR HANDLING
	try {
		<trybody>
	} catch { // If an error occurs in <trybody>
		<catchbody>
	} otherwise { // If <trybody> executed successfully
		<otherwisebody>
	}

### ARRAYS
	// Arrays are static-typed -> they can only contain 1 data type
	// Implicit conversion between integers and reals is also accounted
	[0, 1, 2] -> array[integer]
	[2.718, 3.142, 1.618, 4] -> array[real]
	["Hello", "world!"] -> array[string]
	[true, true, false] -> array[boolean]
	['a', '1', ' ', ''] -> array[char]

	[0, 1] + [2, 3] -> [0, 1, 2, 3] // Appending to arrays
	array[index] // Accessing an array
	array[index] = value // Changing an array's contents

### BUILT-IN FUNCTIONS
	print(*args) // Outputs the arguments to the terminal.
	read() // Prompts for user input.
	error(message: type<string>) // Raises an error with a message.

	to_string(value) // Converts a value to a string.
	to_integer(value) // Converts a value to an integer.
	to_real(value) // Converts a value to a real.
	to_boolean(value) // Converts a value to a boolean.
	to_char(value) // Converts a value to a character.

 	is_string(value) // Checks if a value is a string.
	is_integer(value) // Checks if a value is an integer.
	is_real(value) // Checks if a value to a real.
	is_boolean(value) // Checks if a value is a boolean.
	is_char(value) // Checks if a value is a character.

	open_file(fn: type <string>, mode: type <char>) // Opens a file in either read, write or append mode.
	close_file(file: type <file>) // Closes a file.
	read_file(file: type <file>) // Reads the next line of a file.
	write_file(file: type <file>, content: type <string>) // Writes the content to the file.
	is_eof(file: type <file>) // Checks if end-of-file is reached.

	format(string_to_format: type <string>, *args) // Formats a string.
	pow(a: type <integer/real>, b: type <integer/real>, c: type <integer>) // Returns (a^b) % c or a^b if c is not given.
	type(value) // Returns the value type as a typing object.
	length(value: type <string/array>) // Returns the length of a string or array.
	max(x, y) // Returns the greater value between x and y.
	min(x, y) // Returns the lesser value between x and y.
 	sum(arr: type<array>) // Returns the sum of an array's elements.
	reverse(value: type <string/array>) // Reveses the string/array.
	sorted(arr: type<array>) // Returns a sorted array.

	rand() // Returns a random real x, 0 <= x < 1
	randint(start_val: type <integer>, end_val: type <integer>) // Returns a random integer x, start_val <= x <= end_val

## CHANGE LOG
1.0: Not much
1.1: Added the `sum()`, `is_string()`, `is_integer()`, `is_real()`, `is_boolean()`, and `is_char()` built-in functions.
