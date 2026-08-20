#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include "types.hpp"

namespace ob {
template<std::size_t CAPACITY>
class IdIndex {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "CAPACITY must be a power of two");
    static constexpr std::size_t MASK = CAPACITY - 1;

    struct Entry {
        OrderId key = 0;
        Slot value = INVALID_SLOT;
        bool occupied = false;
    };

    std::array<Entry, CAPACITY> table_{};

    static std::size_t hash(OrderId id) {
        // splitmix64 finalizer — one multiply-heavy mix, good avalanche.
        uint64_t x = id;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x) & MASK;
    }

    static std::size_t dist(std::size_t from , std::size_t to) {
        return (to - from) & MASK;
    }

public:
    bool insert(OrderId id, Slot slot) {
        std::size_t i = hash(id);
        for (std::size_t probes = 0; probes < CAPACITY; ++probes) {
            if (!table_[i].occupied) {
                table_[i] = Entry{id, slot, true};
                return true;
            }
            i = (i+1) & MASK;
        }
        return false;
    }

    Slot find(OrderId id) const {
        std::size_t i = hash(id);
        for (std::size_t probes = 0; probes < CAPACITY; ++probes) {
            const Entry& e = table_[i];
            if (!e.occupied) return INVALID_SLOT;
            if (e.key == id) return e.value;
            i = (i+1) & MASK;
        }
        return INVALID_SLOT;
    }

    void erase(OrderId id) {
        std::size_t i = hash(id);
        std::size_t probes = 0;
        while (table_[i].occupied && table_[i].key != id) {
            i = (i+1) & MASK;
            if (++probes >= CAPACITY) return;
        }
        if (!table_[i].occupied) return;
        table_[i].occupied = false;
        std::size_t hole = i;
        std::size_t j = (i+1) & MASK;
        while(table_[j].occupied) {
            std::size_t home = hash(table_[j].key);
            if(dist(home, hole) <= dist(home, j)) {
                table_[hole] = table_[j];
                table_[j].occupied = false;
                hole = j;
            } 
            j = (j + 1) & MASK;
        }
    }
};
}