# LLVM based numeric specializer in 1000 lines.
from __future__ import print_function

import sys
import ast
import types
import ctypes
import inspect
import pprint
import string
import numpy as np
from itertools import tee, izip

from textwrap import dedent
from collections import deque, defaultdict

import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from llvm.core import Module, Builder, Function, Type, Constant

DEBUG = False

### == Core Language ==

class Var(ast.AST):
    _fields = ["id", "type"]

    def __init__(self, id, type=None):
        self.id = id
        self.type = type

class Assign(ast.AST):
    _fields = ["ref", "val", "type"]

    def __init__(self, ref, val, type=None):
        self.ref = ref
        self.val = val
        self.type = type

class Return(ast.AST):
    _fields = ["val"]

    def __init__(self, val):
        self.val = val

class Loop(ast.AST):
    _fields = ["var", "begin", "end", "body"]

    def __init__(self, var, begin, end, body):
        self.var = var
        self.begin = begin
        self.end = end
        self.body = body

class App(ast.AST):
    _fields = ["fn", "args"]

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

class Fun(ast.AST):
    _fields = ["fname", "args", "body"]

    def __init__(self, fname, args, body):
        self.fname = fname
        self.args = args
        self.body = body

class LitInt(ast.AST):
    _fields = ["n"]

    def __init__(self, n, type=None):
        self.n = n
        self.type = type

class LitFloat(ast.AST):
    _fields = ["n"]

    def __init__(self, n, type=None):
        self.n = n
        self.type = None

class LitBool(ast.AST):
    _fields = ["n"]

    def __init__(self, n):
        self.n = n

class Prim(ast.AST):
    _fields = ["fn", "args"]

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

class Index(ast.AST):
    _fields = ["val", "ix"]

    def __init__(self, val, ix):
        self.val = val
        self.ix = ix

class Noop(ast.AST):
    _fields = []

primops = {ast.Add: "add#", ast.Mult: "mult#"}

### == Type System ==

class TVar(object):
    def __init__(self, s):
        self.s = s

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if isinstance(other, TVar):
            return (self.s == other.s)
        else:
            return False

    def __str__(self):
        return self.s

class TCon(object):
    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        if isinstance(other, TCon):
            return (self.s == other.s)
        else:
            return False

    def __hash__(self):
        return hash(self.s)

    def __str__(self):
        return self.s

class TApp(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if isinstance(other, TApp):
            return (self.a == other.a) & (self.b == other.b)
        else:
            return False

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return str(self.a) + " " + str(self.b)

class TFun(object):
    def __init__(self, argtys, retty):
        assert isinstance(argtys, list)
        self.argtys = argtys
        self.retty = retty

    def __eq__(self, other):
        if isinstance(other, TFun):
            return (self.argtys == other.argtys) & (self.retty == other.retty)
        else:
            return False

    def __str__(self):
        return str(self.argtys) + " -> " + str(self.retty)

def ftv(x):
    if isinstance(x, TCon):
        return set()
    elif isinstance(x, TApp):
        return ftv(x.a) | ftv(x.b)
    elif isinstance(x, TFun):
        return reduce(set.union, map(ftv, x.argtys)) | ftv(x.retty)
    elif isinstance(x, TVar):
        return set([x])

def is_array(ty):
    return isinstance(ty, TApp) and ty.a == TCon("Array")

int32 = TCon("Int32")
int64 = TCon("Int64")
float32 = TCon("Float")
double64 = TCon("Double")
void = TCon("Void")
array = lambda t: TApp(TCon("Array"), t)

array_int32 = array(int32)
array_int64 = array(int64)
array_double64 = array(double64)

### == Type Inference ==

def naming():
    k = 0
    while True:
        for a in string.ascii_lowercase:
            yield ("'"+a+str(k)) if (k > 0) else (a)
        k = k+1

class TypeInfer(object):

    def __init__(self):
        self.constraints = []
        self.env = {}
        self.names = naming()

    def fresh(self):
        return TVar('$' + next(self.names))  # New meta type variable.

    def visit(self, node):
        name = "visit_%s" % type(node).__name__
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

    def visit_Fun(self, node):
        arity = len(node.args)
        self.argtys = [self.fresh() for v in node.args]
        self.retty = TVar("$retty")
        for (arg, ty) in zip(node.args, self.argtys):
            arg.type = ty
            self.env[arg.id] = ty
        map(self.visit, node.body)
        return TFun(self.argtys, self.retty)

    def visit_Noop(self, node):
        return None

    def visit_LitInt(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_LitFloat(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_Assign(self, node):
        ty = self.visit(node.val)
        if node.ref in self.env:
            # Subsequent uses of a variable must have the same type.
            self.constraints += [(ty, self.env[node.ref])]
        self.env[node.ref] = ty
        node.type = ty
        return None

    def visit_Index(self, node):
        tv = self.fresh()
        ty = self.visit(node.val)
        ixty = self.visit(node.ix)
        self.constraints += [(ty, array(tv)), (ixty, int32)]
        return tv

    def visit_Prim(self, node):
        if node.fn == "shape#":
            return array(int32)
        elif node.fn == "mult#":
            tya = self.visit(node.args[0])
            tyb = self.visit(node.args[1])
            self.constraints += [(tya, tyb)]
            return tyb
        elif node.fn == "add#":
            tya = self.visit(node.args[0])
            tyb = self.visit(node.args[1])
            self.constraints += [(tya, tyb)]
            return tyb
        else:
            raise NotImplementedError

    def visit_Var(self, node):
        ty = self.env[node.id]
        node.type = ty
        return ty

    def visit_Return(self, node):
        ty = self.visit(node.val)
        self.constraints += [(ty, self.retty)]

    def visit_Loop(self, node):
        self.env[node.var.id] = int32
        varty = self.visit(node.var)
        begin = self.visit(node.begin)
        end = self.visit(node.end)
        self.constraints += [(varty, int32), (
            begin, int64), (end, int32)]
        map(self.visit, node.body)

    def generic_visit(self, node):
        raise NotImplementedError

class UnderDeteremined(Exception):
    def __str__(self):
        return "The types in the function are not fully determined by the \
                input types. Add annotations."

class InferError(Exception):
    def __init__(self, ty1, ty2):
        self.ty1 = ty1
        self.ty2 = ty2

    def __str__(self):
        return '\n'.join([
            "Type mismatch: ",
            "Given: ", "\t" + str(self.ty1),
            "Expected: ", "\t" + str(self.ty2)
        ])

### == Constraint Solver ==

def empty():
    return {}

def apply(s, t):
    if isinstance(t, TCon):
        return t
    elif isinstance(t, TApp):
        return TApp(apply(s, t.a), apply(s, t.b))
    elif isinstance(t, TFun):
        argtys = [apply(s, a) for a in t.argtys]
        retty = apply(s, t.retty)
        return TFun(argtys, retty)
    elif isinstance(t, TVar):
        return s.get(t.s, t)

def applyList(s, xs):
    return [(apply(s, x), apply(s, y)) for (x, y) in xs]

def unify(x, y):
    if isinstance(x, TApp) and isinstance(y, TApp):
        s1 = unify(x.a, y.a)
        s2 = unify(apply(s1, x.b), apply(s1, y.b))
        return compose(s2, s1)
    elif isinstance(x, TCon) and isinstance(y, TCon) and (x == y):
        return empty()
    elif isinstance(x, TFun) and isinstance(y, TFun):
        if len(x.argtys) != len(y.argtys):
            return Exception("Wrong number of arguments")
        s1 = solve(zip(x.argtys, y.argtys))
        s2 = unify(apply(s1, x.retty), apply(s1, y.retty))
        return compose(s2, s1)
    elif isinstance(x, TVar):
        return bind(x.s, y)
    elif isinstance(y, TVar):
        return bind(y.s, x)
    else:
        raise InferError(x, y)

def solve(xs):
    mgu = empty()
    cs = deque(xs)
    while len(cs):
        (a, b) = cs.pop()
        s = unify(a, b)
        mgu = compose(s, mgu)
        cs = deque(applyList(s, cs))
    return mgu

def bind(n, x):
    if x == n:
        return empty()
    elif occurs_check(n, x):
        raise InfiniteType(n, x)
    else:
        return dict([(n, x)])

def occurs_check(n, x):
    return n in ftv(x)

def union(s1, s2):
    nenv = s1.copy()
    nenv.update(s2)
    return nenv

def compose(s1, s2):
    s3 = dict((t, apply(s1, u)) for t, u in s2.items())
    return union(s1, s3)

### == Core Translator ==

class PythonVisitor(ast.NodeVisitor):

    def __init__(self):
        pass

    def __call__(self, source):
        if isinstance(source, types.ModuleType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.FunctionType):
            source = dedent(inspect.getsource(source))
        if isinstance(source, types.LambdaType):
            source = dedent(inspect.getsource(source))
        elif isinstance(source, (str, unicode)):
            source = dedent(source)
        else:
            raise NotImplementedError

        self._source = source
        self._ast = ast.parse(source)
        return self.visit(self._ast)

    def visit_Module(self, node):
        body = map(self.visit, node.body)
        return body[0]

    def visit_Name(self, node):
        return Var(node.id)

    def visit_Num(self, node):
        if isinstance(node.n, float):
            return LitFloat(node.n)
        else:
            return LitInt(node.n)

    def visit_Bool(self, node):
        return LitBool(node.n)

    def visit_Call(self, node):
        name = self.visit(node.func)
        args = map(self.visit, node.args)
        keywords = map(self.visit, node.keywords)
        return App(name, args)

    def visit_BinOp(self, node):
        op_str = node.op.__class__
        a = self.visit(node.left)
        b = self.visit(node.right)
        opname = primops[op_str]
        return Prim(opname, [a, b])

    def visit_Assign(self, node):
        targets = node.targets

        assert len(node.targets) == 1
        var = node.targets[0].id
        val = self.visit(node.value)
        return Assign(var, val)

    def visit_FunctionDef(self, node):
        stmts = list(node.body)
        stmts = map(self.visit, stmts)
        args = map(self.visit, node.args.args)
        res = Fun(node.name, args, stmts)
        return res

    def visit_Pass(self, node):
        return Noop()

    def visit_Return(self, node):
        val = self.visit(node.value)
        return Return(val)

    def visit_Attribute(self, node):
        if node.attr == "shape":
            val = self.visit(node.value)
            return Prim("shape#", [val])
        else:
            raise NotImplementedError

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.slice:
                val = self.visit(node.value)
                ix = self.visit(node.slice.value)
                return Index(val, ix)
        elif isinstance(node.ctx, ast.Store):
            raise NotImplementedError

    def visit_For(self, node):
        target = self.visit(node.target)
        stmts = map(self.visit, node.body)
        if node.iter.func.id in {"xrange", "range"}:
            args = map(self.visit, node.iter.args)
        else:
            raise Exception("Loop must be over range")

        if len(args) == 1:   # xrange(n)
            return Loop(target, LitInt(0, type=int32), args[0], stmts)
        elif len(args) == 2:  # xrange(n,m)
            return Loop(target, args[0], args[1], stmts)

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim("add#", [Var(ref), value]))
        if isinstance(node.op, ast.Mul):
            ref = node.target.id
            value = self.visit(node.value)
            return Assign(ref, Prim("mult#", [Var(ref), value]))
        else:
            raise NotImplementedError

    def generic_visit(self, node):
        raise NotImplementedError

### == Pretty Printer ==

# From my coworker John Riehl, pretty sure he dont't care.
def ast2tree(node, include_attrs=True):
    def _transform(node):
        if isinstance(node, ast.AST):
            fields = ((a, _transform(b))
                      for a, b in ast.iter_fields(node))
            if include_attrs:
                attrs = ((a, _transform(getattr(node, a)))
                         for a in node._attributes
                         if hasattr(node, a))
                return (node.__class__.__name__, dict(fields), dict(attrs))
            return (node.__class__.__name__, dict(fields))
        elif isinstance(node, list):
            return [_transform(x) for x in node]
        elif isinstance(node, str):
            return repr(node)
        return node
    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _transform(node)

def pformat_ast(node, include_attrs=False, **kws):
    return pprint.pformat(ast2tree(node, include_attrs), **kws)

def dump(node):
    return pformat_ast(node)

### == LLVM Codegen ==

pointer     = Type.pointer
int_type    = Type.int()
float_type  = Type.float()
double_type = Type.double()
bool_type   = Type.int(1)
void_type   = Type.void()
void_ptr    = pointer(Type.int(8))

def array_type(elt_type):
    return Type.struct([
        pointer(elt_type),  # data
        int_type,           # dimensions
        pointer(int_type),  # shape
    ], name='ndarray_' + str(elt_type))

int32_array = pointer(array_type(int_type))
int64_array = pointer(array_type(Type.int(64)))
double_array = pointer(array_type(double_type))

lltypes_map = {
    int32          : int_type,
    int64          : int_type,
    float32        : float_type,
    double64       : double_type,
    array_int32    : int32_array,
    array_int64    : int64_array,
    array_double64 : double_array
}

def to_lltype(ptype):
    return lltypes_map[ptype]

def determined(ty):
    return len(ftv(ty)) == 0

class LLVMEmitter(object):
    def __init__(self, spec_types, retty, argtys):
        self.function = None             # LLVM Function
        self.builder = None              # LLVM Builder
        self.locals = {}                 # Local variables
        self.arrays = defaultdict(dict)  # Array metadata
        self.exit_block = None           # Exit block
        self.spec_types = spec_types     # Type specialization
        self.retty = retty               # Return type
        self.argtys = argtys             # Argument types

    def start_function(self, name, module, rettype, argtypes):
        func_type = lc.Type.function(rettype, argtypes, False)
        function = lc.Function.new(module, func_type, name)
        entry_block = function.append_basic_block("entry")
        builder = lc.Builder.new(entry_block)
        self.exit_block = function.append_basic_block("exit")
        self.function = function
        self.builder = builder

    def end_function(self):
        self.builder.position_at_end(self.exit_block)

        if 'retval' in self.locals:
            retval = self.builder.load(self.locals['retval'])
            self.builder.ret(retval)
        else:
            self.builder.ret_void()

    def add_block(self, name):
        return self.function.append_basic_block(name)

    def set_block(self, block):
        self.block = block
        self.builder.position_at_end(block)

    def cbranch(self, cond, true_block, false_block):
        self.builder.cbranch(cond, true_block, false_block)

    def branch(self, next_block):
        self.builder.branch(next_block)

    def specialize(self, val):
        if isinstance(val.type, TVar):
            return to_lltype(self.spec_types[val.type.s])
        else:
            return val.type

    def const(self, val):
        if isinstance(val, (int, long)):
            return Constant.int(int_type, val)
        elif isinstance(val, float):
            return Constant.real(double_type, val)
        elif isinstance(val, bool):
            return Constant.int(bool_type, int(val))
        elif isinstance(val, str):
            return Constant.stringz(val)
        else:
            raise NotImplementedError

    def visit_LitInt(self, node):
        ty = self.specialize(node)
        if ty is double_type:
            return Constant.real(double_type, node.n)
        elif ty == int_type:
            return Constant.int(int_type, node.n)

    def visit_LitFloat(self, node):
        ty = self.specialize(node)
        if ty is double_type:
            return Constant.real(double_type, node.n)
        elif ty == int_type:
            return Constant.int(int_type, node.n)

    def visit_Noop(self, node):
        pass

    def visit_Fun(self, node):
        rettype = to_lltype(self.retty)
        argtypes = map(to_lltype, self.argtys)
        # Create a unique specialized name
        func_name = mangler(node.fname, self.argtys)
        self.start_function(func_name, module, rettype, argtypes)

        for (ar, llarg, argty) in zip(node.args, self.function.args, self.argtys):
            name = ar.id
            llarg.name = name

            if is_array(argty):
                zero = self.const(0)
                one = self.const(1)
                two = self.const(2)

                data = self.builder.gep(llarg, [
                                        zero, zero], name=(name + '_data'))
                dims = self.builder.gep(llarg, [
                                        zero, one], name=(name + '_dims'))
                shape = self.builder.gep(llarg, [
                                         zero, two], name=(name + '_strides'))

                self.arrays[name]['data'] = self.builder.load(data)
                self.arrays[name]['dims'] = self.builder.load(dims)
                self.arrays[name]['shape'] = self.builder.load(shape)
                self.locals[name] = llarg
            else:
                argref = self.builder.alloca(to_lltype(argty))
                self.builder.store(llarg, argref)
                self.locals[name] = argref

        # Setup the register for return type.
        if rettype is not void_type:
            self.locals['retval'] = self.builder.alloca(rettype, name="retval")

        map(self.visit, node.body)
        self.end_function()

    def visit_Index(self, node):
        if isinstance(node.val, Var) and node.val.id in self.arrays:
            val = self.visit(node.val)
            ix = self.visit(node.ix)
            dataptr = self.arrays[node.val.id]['data']
            ret = self.builder.gep(dataptr, [ix])
            return self.builder.load(ret)
        else:
            val = self.visit(node.val)
            ix = self.visit(node.ix)
            ret = self.builder.gep(val, [ix])
            return self.builder.load(ret)

    def visit_Var(self, node):
        return self.builder.load(self.locals[node.id])

    def visit_Return(self, node):
        val = self.visit(node.val)
        if val.type != void_type:
            self.builder.store(val, self.locals['retval'])
        self.builder.branch(self.exit_block)

    def visit_Loop(self, node):
        init_block = self.function.append_basic_block('for.init')
        test_block = self.function.append_basic_block('for.cond')
        body_block = self.function.append_basic_block('for.body')
        end_block = self.function.append_basic_block("for.end")

        self.branch(init_block)
        self.set_block(init_block)

        start = self.visit(node.begin)
        stop = self.visit(node.end)
        step = 1

        # Setup the increment variable
        varname = node.var.id
        inc = self.builder.alloca(int_type, name=varname)
        self.builder.store(start, inc)
        self.locals[varname] = inc

        # Setup the loop condition
        self.branch(test_block)
        self.set_block(test_block)
        cond = self.builder.icmp(lc.ICMP_SLT, self.builder.load(inc), stop)
        self.builder.cbranch(cond, body_block, end_block)

        # Generate the loop body
        self.set_block(body_block)
        map(self.visit, node.body)

        # Increment the counter
        succ = self.builder.add(self.const(step), self.builder.load(inc))
        self.builder.store(succ, inc)

        # Exit the loop
        self.builder.branch(test_block)
        self.set_block(end_block)

    def visit_Prim(self, node):
        if node.fn == "shape#":
            ref = node.args[0]
            shape = self.arrays[ref.id]['shape']
            return shape
        elif node.fn == "mult#":
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == double_type:
                return self.builder.fmul(a, b)
            else:
                return self.builder.mul(a, b)
        elif node.fn == "add#":
            a = self.visit(node.args[0])
            b = self.visit(node.args[1])
            if a.type == double_type:
                return self.builder.fadd(a, b)
            else:
                return self.builder.add(a, b)
        else:
            raise NotImplementedError

    def visit_Assign(self, node):
        # Subsequent assignment
        if node.ref in self.locals:
            name = node.ref
            var = self.locals[name]
            val = self.visit(node.val)
            self.builder.store(val, var)
            self.locals[name] = var
            return var

        # First assignment
        else:
            name = node.ref
            val = self.visit(node.val)
            ty = self.specialize(node)
            var = self.builder.alloca(ty, name=name)
            self.builder.store(val, var)
            self.locals[name] = var
            return var

    def visit(self, node):
        name = "visit_%s" % type(node).__name__
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

### == Type Mapping ==

# Adapt the LLVM types to use libffi/ctypes wrapper so we can dynamically create
# the appropriate C types for our JIT'd function at runtime.
_nptypemap = {
    'i': ctypes.c_int,
    'f': ctypes.c_float,
    'd': ctypes.c_double,
}

def mangler(fname, sig):
    return fname + str(hash(tuple(sig)))

def wrap_module(sig, llfunc):
    pfunc = wrap_function(llfunc, engine)
    dispatch = dispatcher(pfunc)
    return dispatch

def wrap_function(func, engine):
    args = func.type.pointee.args
    ret_type = func.type.pointee.return_type
    ret_ctype = wrap_type(ret_type)
    args_ctypes = map(wrap_type, args)

    functype = ctypes.CFUNCTYPE(ret_ctype, *args_ctypes)
    fptr = engine.get_pointer_to_function(func)

    cfunc = functype(fptr)
    cfunc.__name__ = func.name
    return cfunc

def wrap_type(llvm_type):
    kind = llvm_type.kind
    if kind == lc.TYPE_INTEGER:
        ctype = getattr(ctypes, "c_int"+str(llvm_type.width))
    elif kind == lc.TYPE_DOUBLE:
        ctype = ctypes.c_double
    elif kind == lc.TYPE_FLOAT:
        ctype = ctypes.c_float
    elif kind == lc.TYPE_VOID:
        ctype = None
    elif kind == lc.TYPE_POINTER:
        pointee = llvm_type.pointee
        p_kind = pointee.kind
        if p_kind == lc.TYPE_INTEGER:
            width = pointee.width
            if width == 8:
                ctype = ctypes.c_char_p
            else:
                ctype = ctypes.POINTER(wrap_type(pointee))
        elif p_kind == lc.TYPE_VOID:
            ctype = ctypes.c_void_p
        else:
            ctype = ctypes.POINTER(wrap_type(pointee))
    elif kind == lc.TYPE_STRUCT:
        struct_name = llvm_type.name.split('.')[-1]
        struct_name = struct_name.encode('ascii')
        struct_type = None

        if struct_type and issubclass(struct_type, ctypes.Structure):
            return struct_type

        if hasattr(struct_type, '_fields_'):
            names = struct_type._fields_
        else:
            names = ["field"+str(n) for n in range(llvm_type.element_count)]

        ctype = type(ctypes.Structure)(struct_name, (ctypes.Structure,),
                                       {'__module__': "numpile"})

        fields = [(name, wrap_type(elem))
                  for name, elem in zip(names, llvm_type.elements)]
        setattr(ctype, '_fields_', fields)
    else:
        raise Exception("Unknown LLVM type %s" % kind)
    return ctype

def wrap_ndarray(na):
    # For NumPy arrays grab the underlying data pointer. Doesn't copy.
    ctype = _nptypemap[na.dtype.char]
    _shape = list(na.shape)
    data = na.ctypes.data_as(ctypes.POINTER(ctype))
    dims = len(na.strides)
    shape = (ctypes.c_int*dims)(*_shape)
    return (data, dims, shape)

def wrap_arg(arg, val):
    if isinstance(val, np.ndarray):
        ndarray = arg._type_
        data, dims, shape = wrap_ndarray(val)
        return ndarray(data, dims, shape)
    else:
        return val

def dispatcher(fn):
    def _call_closure(*args):
        cargs = list(fn._argtypes_)
        pargs = list(args)
        rargs = map(wrap_arg, cargs, pargs)
        return fn(*rargs)
    _call_closure.__name__ = fn.__name__
    return _call_closure

### == Toplevel ==

module = lc.Module.new('numpile.module')
engine = None
function_cache = {}

tm = le.TargetMachine.new(features='', cm=le.CM_JITDEFAULT)
eb = le.EngineBuilder.new(module)
engine = eb.create(tm)

def autojit(fn):
    transformer = PythonVisitor()
    ast = transformer(fn)
    (ty, mgu) = typeinfer(ast)
    debug(dump(ast))
    return specialize(ast, ty, mgu)

def arg_pytype(arg):
    if isinstance(arg, np.ndarray):
        if arg.dtype == np.dtype('int32'):
            return array(int32)
        elif arg.dtype == np.dtype('int64'):
            return array(int64)
        elif arg.dtype == np.dtype('double'):
            return array(double64)
        elif arg.dtype == np.dtype('float'):
            return array(float32)
    elif isinstance(arg, int) & (arg < sys.maxint):
        return int64
    elif isinstance(arg, float):
        return double64
    else:
        raise Exception("Type not supported: %s" % type(arg))

def specialize(ast, infer_ty, mgu):
    def _wrapper(*args):
        types = map(arg_pytype, list(args))
        spec_ty = TFun(argtys=types, retty=TVar("$retty"))
        unifier = unify(infer_ty, spec_ty)
        specializer = compose(unifier, mgu)

        retty = apply(specializer, TVar("$retty"))
        argtys = [apply(specializer, ty) for ty in types]
        debug('Specialized Function:', TFun(argtys, retty))

        if determined(retty) and all(map(determined, argtys)):
            key = mangler(ast.fname, argtys)
            # Don't recompile after we've specialized.
            if key in function_cache:
                return function_cache[key](*args)
            else:
                llfunc = codegen(ast, specializer, retty, argtys)
                pyfunc = wrap_module(argtys, llfunc)
                function_cache[key] = pyfunc
                return pyfunc(*args)
        else:
            raise UnderDeteremined()
    return _wrapper

def typeinfer(ast):
    infer = TypeInfer()
    ty = infer.visit(ast)
    mgu = solve(infer.constraints)
    infer_ty = apply(mgu, ty)
    debug(infer_ty)
    debug(mgu)
    debug(infer.constraints)
    return (infer_ty, mgu)

def codegen(ast, specializer, retty, argtys):
    cgen = LLVMEmitter(specializer, retty, argtys)
    mod = cgen.visit(ast)
    cgen.function.verify()

    tm = le.TargetMachine.new(opt=3, cm=le.CM_JITDEFAULT, features='')
    pms = lp.build_pass_managers(tm=tm,
                                 fpm=False,
                                 mod=module,
                                 opt=3,
                                 vectorize=False,
                                 loop_vectorize=True)
    pms.pm.run(module)

    debug(cgen.function)
    debug(module.to_native_assembly())
    return cgen.function

def debug(fmt, *args):
    if DEBUG:
        print('=' * 80)
        print(fmt, *args)
