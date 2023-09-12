# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.0
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _envcpp
else:
    import _envcpp

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _envcpp.delete_SwigPyIterator

    def value(self):
        return _envcpp.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _envcpp.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _envcpp.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _envcpp.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _envcpp.SwigPyIterator_equal(self, x)

    def copy(self):
        return _envcpp.SwigPyIterator_copy(self)

    def next(self):
        return _envcpp.SwigPyIterator_next(self)

    def __next__(self):
        return _envcpp.SwigPyIterator___next__(self)

    def previous(self):
        return _envcpp.SwigPyIterator_previous(self)

    def advance(self, n):
        return _envcpp.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _envcpp.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _envcpp.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _envcpp.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _envcpp.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _envcpp.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _envcpp.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _envcpp:
_envcpp.SwigPyIterator_swigregister(SwigPyIterator)
class vectori(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _envcpp.vectori_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _envcpp.vectori___nonzero__(self)

    def __bool__(self):
        return _envcpp.vectori___bool__(self)

    def __len__(self):
        return _envcpp.vectori___len__(self)

    def __getslice__(self, i, j):
        return _envcpp.vectori___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _envcpp.vectori___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _envcpp.vectori___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _envcpp.vectori___delitem__(self, *args)

    def __getitem__(self, *args):
        return _envcpp.vectori___getitem__(self, *args)

    def __setitem__(self, *args):
        return _envcpp.vectori___setitem__(self, *args)

    def pop(self):
        return _envcpp.vectori_pop(self)

    def append(self, x):
        return _envcpp.vectori_append(self, x)

    def empty(self):
        return _envcpp.vectori_empty(self)

    def size(self):
        return _envcpp.vectori_size(self)

    def swap(self, v):
        return _envcpp.vectori_swap(self, v)

    def begin(self):
        return _envcpp.vectori_begin(self)

    def end(self):
        return _envcpp.vectori_end(self)

    def rbegin(self):
        return _envcpp.vectori_rbegin(self)

    def rend(self):
        return _envcpp.vectori_rend(self)

    def clear(self):
        return _envcpp.vectori_clear(self)

    def get_allocator(self):
        return _envcpp.vectori_get_allocator(self)

    def pop_back(self):
        return _envcpp.vectori_pop_back(self)

    def erase(self, *args):
        return _envcpp.vectori_erase(self, *args)

    def __init__(self, *args):
        _envcpp.vectori_swiginit(self, _envcpp.new_vectori(*args))

    def push_back(self, x):
        return _envcpp.vectori_push_back(self, x)

    def front(self):
        return _envcpp.vectori_front(self)

    def back(self):
        return _envcpp.vectori_back(self)

    def assign(self, n, x):
        return _envcpp.vectori_assign(self, n, x)

    def resize(self, *args):
        return _envcpp.vectori_resize(self, *args)

    def insert(self, *args):
        return _envcpp.vectori_insert(self, *args)

    def reserve(self, n):
        return _envcpp.vectori_reserve(self, n)

    def capacity(self):
        return _envcpp.vectori_capacity(self)
    __swig_destroy__ = _envcpp.delete_vectori

# Register vectori in _envcpp:
_envcpp.vectori_swigregister(vectori)
class vectord(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _envcpp.vectord_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _envcpp.vectord___nonzero__(self)

    def __bool__(self):
        return _envcpp.vectord___bool__(self)

    def __len__(self):
        return _envcpp.vectord___len__(self)

    def __getslice__(self, i, j):
        return _envcpp.vectord___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _envcpp.vectord___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _envcpp.vectord___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _envcpp.vectord___delitem__(self, *args)

    def __getitem__(self, *args):
        return _envcpp.vectord___getitem__(self, *args)

    def __setitem__(self, *args):
        return _envcpp.vectord___setitem__(self, *args)

    def pop(self):
        return _envcpp.vectord_pop(self)

    def append(self, x):
        return _envcpp.vectord_append(self, x)

    def empty(self):
        return _envcpp.vectord_empty(self)

    def size(self):
        return _envcpp.vectord_size(self)

    def swap(self, v):
        return _envcpp.vectord_swap(self, v)

    def begin(self):
        return _envcpp.vectord_begin(self)

    def end(self):
        return _envcpp.vectord_end(self)

    def rbegin(self):
        return _envcpp.vectord_rbegin(self)

    def rend(self):
        return _envcpp.vectord_rend(self)

    def clear(self):
        return _envcpp.vectord_clear(self)

    def get_allocator(self):
        return _envcpp.vectord_get_allocator(self)

    def pop_back(self):
        return _envcpp.vectord_pop_back(self)

    def erase(self, *args):
        return _envcpp.vectord_erase(self, *args)

    def __init__(self, *args):
        _envcpp.vectord_swiginit(self, _envcpp.new_vectord(*args))

    def push_back(self, x):
        return _envcpp.vectord_push_back(self, x)

    def front(self):
        return _envcpp.vectord_front(self)

    def back(self):
        return _envcpp.vectord_back(self)

    def assign(self, n, x):
        return _envcpp.vectord_assign(self, n, x)

    def resize(self, *args):
        return _envcpp.vectord_resize(self, *args)

    def insert(self, *args):
        return _envcpp.vectord_insert(self, *args)

    def reserve(self, n):
        return _envcpp.vectord_reserve(self, n)

    def capacity(self):
        return _envcpp.vectord_capacity(self)
    __swig_destroy__ = _envcpp.delete_vectord

# Register vectord in _envcpp:
_envcpp.vectord_swigregister(vectord)
class vectors(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _envcpp.vectors_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _envcpp.vectors___nonzero__(self)

    def __bool__(self):
        return _envcpp.vectors___bool__(self)

    def __len__(self):
        return _envcpp.vectors___len__(self)

    def __getslice__(self, i, j):
        return _envcpp.vectors___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _envcpp.vectors___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _envcpp.vectors___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _envcpp.vectors___delitem__(self, *args)

    def __getitem__(self, *args):
        return _envcpp.vectors___getitem__(self, *args)

    def __setitem__(self, *args):
        return _envcpp.vectors___setitem__(self, *args)

    def pop(self):
        return _envcpp.vectors_pop(self)

    def append(self, x):
        return _envcpp.vectors_append(self, x)

    def empty(self):
        return _envcpp.vectors_empty(self)

    def size(self):
        return _envcpp.vectors_size(self)

    def swap(self, v):
        return _envcpp.vectors_swap(self, v)

    def begin(self):
        return _envcpp.vectors_begin(self)

    def end(self):
        return _envcpp.vectors_end(self)

    def rbegin(self):
        return _envcpp.vectors_rbegin(self)

    def rend(self):
        return _envcpp.vectors_rend(self)

    def clear(self):
        return _envcpp.vectors_clear(self)

    def get_allocator(self):
        return _envcpp.vectors_get_allocator(self)

    def pop_back(self):
        return _envcpp.vectors_pop_back(self)

    def erase(self, *args):
        return _envcpp.vectors_erase(self, *args)

    def __init__(self, *args):
        _envcpp.vectors_swiginit(self, _envcpp.new_vectors(*args))

    def push_back(self, x):
        return _envcpp.vectors_push_back(self, x)

    def front(self):
        return _envcpp.vectors_front(self)

    def back(self):
        return _envcpp.vectors_back(self)

    def assign(self, n, x):
        return _envcpp.vectors_assign(self, n, x)

    def resize(self, *args):
        return _envcpp.vectors_resize(self, *args)

    def insert(self, *args):
        return _envcpp.vectors_insert(self, *args)

    def reserve(self, n):
        return _envcpp.vectors_reserve(self, n)

    def capacity(self):
        return _envcpp.vectors_capacity(self)
    __swig_destroy__ = _envcpp.delete_vectors

# Register vectors in _envcpp:
_envcpp.vectors_swigregister(vectors)
class Environment(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, filedir):
        _envcpp.Environment_swiginit(self, _envcpp.new_Environment(filedir))
    __swig_destroy__ = _envcpp.delete_Environment

    def get_download_time(self, video_chunk_size):
        return _envcpp.Environment_get_download_time(self, video_chunk_size)

    def reset_download_time(self):
        return _envcpp.Environment_reset_download_time(self)

    def get_video_chunk(self, quality):
        return _envcpp.Environment_get_video_chunk(self, quality)

    def get_optimal(self, last_video_vmaf):
        return _envcpp.Environment_get_optimal(self, last_video_vmaf)
    optimal = property(_envcpp.Environment_optimal_get, _envcpp.Environment_optimal_set)
    delay0 = property(_envcpp.Environment_delay0_get, _envcpp.Environment_delay0_set)
    sleep_time0 = property(_envcpp.Environment_sleep_time0_get, _envcpp.Environment_sleep_time0_set)
    return_buffer_size0 = property(_envcpp.Environment_return_buffer_size0_get, _envcpp.Environment_return_buffer_size0_set)
    rebuf0 = property(_envcpp.Environment_rebuf0_get, _envcpp.Environment_rebuf0_set)
    video_chunk_size0 = property(_envcpp.Environment_video_chunk_size0_get, _envcpp.Environment_video_chunk_size0_set)
    end_of_video0 = property(_envcpp.Environment_end_of_video0_get, _envcpp.Environment_end_of_video0_set)
    video_chunk_remain0 = property(_envcpp.Environment_video_chunk_remain0_get, _envcpp.Environment_video_chunk_remain0_set)
    video_chunk_vmaf0 = property(_envcpp.Environment_video_chunk_vmaf0_get, _envcpp.Environment_video_chunk_vmaf0_set)
    all_cooked_bw = property(_envcpp.Environment_all_cooked_bw_get, _envcpp.Environment_all_cooked_bw_set)
    all_cooked_time = property(_envcpp.Environment_all_cooked_time_get, _envcpp.Environment_all_cooked_time_set)
    CHUNK_COMBO_OPTIONS = property(_envcpp.Environment_CHUNK_COMBO_OPTIONS_get, _envcpp.Environment_CHUNK_COMBO_OPTIONS_set)
    all_file_names = property(_envcpp.Environment_all_file_names_get, _envcpp.Environment_all_file_names_set)
    video_chunk_counter = property(_envcpp.Environment_video_chunk_counter_get, _envcpp.Environment_video_chunk_counter_set)
    buffer_size = property(_envcpp.Environment_buffer_size_get, _envcpp.Environment_buffer_size_set)
    trace_idx = property(_envcpp.Environment_trace_idx_get, _envcpp.Environment_trace_idx_set)
    cooked_time = property(_envcpp.Environment_cooked_time_get, _envcpp.Environment_cooked_time_set)
    cooked_bw = property(_envcpp.Environment_cooked_bw_get, _envcpp.Environment_cooked_bw_set)
    mahimahi_start_ptr = property(_envcpp.Environment_mahimahi_start_ptr_get, _envcpp.Environment_mahimahi_start_ptr_set)
    mahimahi_ptr = property(_envcpp.Environment_mahimahi_ptr_get, _envcpp.Environment_mahimahi_ptr_set)
    last_mahimahi_time = property(_envcpp.Environment_last_mahimahi_time_get, _envcpp.Environment_last_mahimahi_time_set)
    virtual_mahimahi_ptr = property(_envcpp.Environment_virtual_mahimahi_ptr_get, _envcpp.Environment_virtual_mahimahi_ptr_set)
    virtual_last_mahimahi_time = property(_envcpp.Environment_virtual_last_mahimahi_time_get, _envcpp.Environment_virtual_last_mahimahi_time_set)

# Register Environment in _envcpp:
_envcpp.Environment_swigregister(Environment)

