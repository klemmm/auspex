import pytest
import config
import numpy as np
from domain import KSetsDomain
from utils import TOP, BOTTOM, Address, Value, AbstractValue
from miasm.expression.expression import ExprId

class TestKSetsDomain:
    def setup_method(self):
        """Set up a fresh domain for each test."""
        self.domain = KSetsDomain(False)

    def test_initial_state(self):
        """Test that the initial state is correctly set up."""
        assert not self.domain.is_bot
        assert self.domain.reg == {ExprId("RSP", 64): {np.uint64(config.STACK_TOP_ADDR)},
                                   ExprId("RBP", 64): {np.uint64(config.STACK_TOP_ADDR)}}
        assert self.domain.mem == {}

    def test_bottom_state(self):
        """Test that the bottom state is correctly handled."""
        bottom_domain = KSetsDomain(True)
        assert bottom_domain.is_bot
        assert bottom_domain.reg == {}
        assert bottom_domain.mem == {}

    def test_join_with_bottom(self):
        """Test joining with the bottom element."""
        bottom_domain = KSetsDomain(True)
        self.domain.join(bottom_domain)
        assert not self.domain.is_bot  # Should remain unchanged

        # Join bottom with non-bottom
        bottom_domain.join(self.domain)
        assert not bottom_domain.is_bot
        assert bottom_domain.equals(self.domain)

    def test_join_values(self):
        """Test joining abstract values."""
        val1 = {np.uint64(1)}
        val2 = {np.uint64(2)}
        
        # Test joining two concrete values
        self.domain.reg[ExprId("RAX", 64)] = val1
        other = KSetsDomain(False)
        other.reg[ExprId("RAX", 64)] = val2
        
        self.domain.join(other)
        assert self.domain.reg[ExprId("RAX", 64)] == {np.uint64(1), np.uint64(2)}

    def test_join_with_top(self):
        """Test joining with TOP."""
        self.domain.reg[ExprId("RAX", 64)] = {np.uint64(1)}
        other = KSetsDomain(False)
        other.reg[ExprId("RAX", 64)] = TOP
        
        self.domain.join(other)
        assert ExprId("RAX", 64) not in self.domain.reg

    def test_strong_update(self):
        """Test strong update of memory."""
        addr = np.uint64(0x1000)
        val = {np.uint32(42)}
        self.domain.strong_update(addr, val, 4)
        assert self.domain.mem[addr] == val

    def test_weak_update(self):
        """Test weak update of memory."""
        addr = np.uint64(0x1000)
        val = {np.uint32(42)}        
        self.domain.weak_update(addr, val, 4)
        assert addr not in self.domain.mem
        

    def test_read(self):
        """Test reading from memory."""
        addr = np.uint64(0x1000)
        val = {np.uint32(42)}
        self.domain.strong_update(addr, val, 4)
        assert self.domain.read(addr, 4) == val
        assert self.domain.read(addr, 8) is TOP  # Different size should return TOP

    def test_equals(self):
        """Test equality comparison."""
        other = KSetsDomain(False)
        assert self.domain.equals(other)
        
        self.domain.reg[ExprId("RAX", 64)] = {np.uint64(1)}
        assert not self.domain.equals(other)
        
        other.reg[ExprId("RAX", 64)] = {np.uint64(1)}
        assert self.domain.equals(other)

    def test_clone(self):
        """Test domain cloning."""
        self.domain.reg[ExprId("RAX", 64)] = {np.uint64(1)}
        clone = self.domain.clone()
        assert clone.equals(self.domain)
        assert clone is not self.domain  # Should be a deep copy

    def test_incompatible_loc(self):
        """Test location compatibility checking."""
        loc = 0x1000
        assert not self.domain.incompatible_loc(loc)
        
        self.domain.reg[ExprId("IRDst", 64)] = {np.uint64(0x2000)}
        assert self.domain.incompatible_loc(loc) 