import ast
import functools
import inspect

class LoopBreaker(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

    def visit_For(self, node):
        new_body = node.body + [ast.Break()]
        node.body = new_body
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        new_body = node.body + [ast.Break()]
        node.body = new_body
        self.generic_visit(node)
        return node

    def visit_AsyncFor(self, node):
        new_body = node.body + [ast.Break()]
        node.body = new_body
        self.generic_visit(node)
        return node


def analyze_function(target_function_name, processing_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                source = inspect.getsource(func)
                tree = ast.parse(source)

                func_def = None
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_def = node
                        break
                else:
                    print(f"Decorator: Could not find FunctionDef for {func.__name__}")
                    return func(*args, **kwargs)

                original_decorators = func_def.decorator_list
                func_def.decorator_list = []
                # we remove the decorators from the functions
                # when the modified functions is called, the decorator is not executed

                class FunctionAnalyzer(ast.NodeVisitor):
                    def __init__(self):
                        self.first_arg_value = None
                        self.local_vars_names = set()  # Track local variable *names*

                    def visit_Assign(self, node):
                        # Capture all assigned variables in the function
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.local_vars_names.add(target.id)
                        self.generic_visit(node)

                    def visit_Call(self, node):
                        if isinstance(node.func, ast.Attribute):
                            full_name = self._get_full_name(node.func)
                            if full_name == target_function_name:
                                if node.args:
                                    first_arg = node.args[0]
                                    try:
                                        if isinstance(first_arg, ast.Name):
                                            # this branch should be executed
                                            if first_arg.id in captured_locals:
                                                self.first_arg_value = captured_locals[
                                                    first_arg.id
                                                ]
                                            else:
                                                print(f"Could not find: {first_arg.id}")
                                        elif isinstance(first_arg, ast.Attribute):
                                            if (
                                                isinstance(first_arg.value, ast.Name)
                                                and first_arg.value.id == "self"
                                            ):
                                                if len(args) > 0:
                                                    self.first_arg_value = getattr(
                                                        args[0], first_arg.attr
                                                    )
                                        elif isinstance(first_arg, ast.Constant):
                                            self.first_arg_value = (
                                                first_arg.value
                                            )
                                    except Exception as e:
                                        print(
                                            f"Decorator: Error during argument evaluation: {e}"
                                        )
                        self.generic_visit(node)

                    def _get_full_name(self, node):
                        if isinstance(node, ast.Attribute):
                            return f"{self._get_full_name(node.value)}.{node.attr}"
                        elif isinstance(node, ast.Name):
                            return node.id
                        return None

                # --- 1. Break Loops ---
                interceptor = LoopBreaker()
                modified_tree = interceptor.visit(tree)
                modified_tree = ast.fix_missing_locations(modified_tree)

                # --- 2. Prepare Execution Context ---
                global_vars = func.__globals__.copy()
                local_vars = {}

                signature = inspect.signature(func)
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                local_vars.update(bound_args.arguments)

                # --- Capture Local Variables During Execution ---
                captured_locals = {}  # Store captured local variables

                def capture_locals(name, value):
                    """Helper function to capture local variables."""
                    captured_locals[name] = value
                    return value

                global_vars["capture_locals"] = capture_locals  # Inject

                # --- Inject capture_locals calls into the AST ---
                class InjectCapture(ast.NodeTransformer):
                    def visit_Assign(self, node):
                        new_nodes = []
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                # Wrap the right-hand side of the assignment with capture_locals
                                capture_call = ast.Call(
                                    func=ast.Name(id="capture_locals", ctx=ast.Load()),
                                    args=[ast.Constant(value=target.id), node.value],
                                    keywords=[],
                                )
                                new_assign = ast.Assign(
                                    targets=[target], value=capture_call
                                )
                                new_nodes.append(new_assign)
                            else:
                                new_nodes.append(node)
                        return new_nodes

                injector = InjectCapture()
                modified_tree = injector.visit(modified_tree)
                modified_tree = ast.fix_missing_locations(
                    modified_tree
                )

                # --- 3. Compile and Execute ---
                compiled_code = compile(modified_tree, "<string>", "exec")
                exec(compiled_code, global_vars, local_vars)

                # --- 4. Call the Modified Function ---
                modified_func = local_vars[func.__name__]
                # we remove the function name from the local vars
                # only the arguments to the decorated function should be in there
                local_vars.pop(func.__name__)
                result = modified_func(**local_vars)

                # --- 5. Analyze and Process ---
                analyzer = FunctionAnalyzer()
                analyzer.visit(modified_tree) # we analyze the tree to intercept
                # the variable that we want
                
                if analyzer.first_arg_value is not None:
                    # the found variable is used
                    value = processing_func(analyzer.first_arg_value)
                    value.reverse()
                    print("here there is the forward pass definition:")
                    print("\n".join(func for func in value))

                return result

            except (IndentationError, TypeError, SyntaxError) as e:
                print(f"Decorator: Could not analyze or execute {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator

