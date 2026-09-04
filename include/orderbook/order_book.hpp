#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include "id_index.hpp"
#include "types.hpp"

namespace ob {

// A fixed-range, array-indexed limit order book for one instrument.
//
// Levels are addressed by tick offset from `base_price`, so a price must
// satisfy base_price <= price < base_price + NUM_LEVELS to be representable
// — the book covers a fixed window, it does not auto-rebase if the market
// walks out of range (see README for why, and what rebasing would take).
//
// The design deliberately avoids the "sorted map keyed by price" approach:
// a red-black tree gives O(log n) with pointer chasing and rebalancing,
// both of which are unpredictable-latency operations. Here, add/cancel/
// modify are O(1), and best_bid/best_ask are a hardware bit-scan over a
// small occupancy bitmap — no pointer chasing, no rebalancing, no per-call
// heap traffic after construction.

template <std::size_t NUM_LEVELS, std::size_t MAX_ORDERS>
class OrderBook {
    static_assert(NUM_LEVELS % 64 == 0, "NUM_LEVELS must be a multiple of 64 for whole bitset words");
    static_assert(MAX_ORDERS > 0 && MAX_ORDERS < (std::size_t{1} << 31), "MAX_ORDERS out of sane range");

    static constexpr std::size_t WORDS = NUM_LEVELS / 64;

    static constexpr std::size_t next_pow2(std::size_t n) {
        std::size_t c = 1;
        while (c < n) c <<= 1;
        return c;
    }
    
    // Keep the id index at <=50% load factor for short expected probe chains.
    static constexpr std::size_t TABLE_CAPACITY = next_pow2(MAX_ORDERS * 2);

    std::array<PriceLevel, NUM_LEVELS> bid_levels_{};
    std::array<PriceLevel, NUM_LEVELS> ask_levels_{};
    std::array<uint64_t, WORDS> bid_occupied_{};
    std::array<uint64_t, WORDS> ask_occupied_{};

    std::array<Order, MAX_ORDERS> arena_{};
    std::array<Slot, MAX_ORDERS> free_stack_{};
    Slot free_top_ = 0;

    IdIndex<TABLE_CAPACITY> id_index_{};

    Price base_price_;

    static std::size_t level_of(Price base, Price p) {
        return static_cast<std::size_t>(p - base); 
    }

    static void set_bit(std::array<uint64_t, WORDS>& words, std::size_t i) {
        words[i >> 6] |= (uint64_t{1} << (i & 63));
    }
    static void clear_bit(std::array<uint64_t, WORDS>& words, std::size_t i) {
        words[i >> 6] &= ~(uint64_t{1} << (i & 63));
    }

public:
    explicit OrderBook(Price base_price) : base_price_(base_price) {
        for (Slot s = 0; s < MAX_ORDERS; ++s) free_stack_[s] = s;
        free_top_ = static_cast<Slot>(MAX_ORDERS);
    }

    bool add_order(OrderId id, Side side, Price price, Qty qty) {
        std::size_t lvl = level_of(base_price_, price);
        if (lvl >= NUM_LEVELS || free_top_ == 0) return false;

        Slot s = free_stack_[--free_top_];
        Order& o = arena_[s];
        o = Order{id, price, qty, INVALID_SLOT, INVALID_SLOT, static_cast<uint32_t>(lvl), side, true};

        auto& levels   = (side == Side::Bid) ? bid_levels_ : ask_levels_;
        auto& occupied = (side == Side::Bid) ? bid_occupied_ : ask_occupied_;
        PriceLevel& level = levels[lvl];

        if (level.tail == INVALID_SLOT) {
            level.head = level.tail = s;
        } else {
            arena_[level.tail].next = s;
            o.prev = level.tail;
            level.tail = s;
        }
        
        if (level.total_qty == 0) set_bit(occupied, lvl);
        level.total_qty += qty;
        level.order_count += 1;

        id_index_.insert(id, s);
        return true;
    }

    bool cancel_order(OrderId id) {
        Slot s = id_index_.find(id);
        if (s == INVALID_SLOT) return false;
        Order& o = arena_[s];

        auto& levels   = (o.side == Side::Bid) ? bid_levels_ : ask_levels_;
        auto& occupied = (o.side == Side::Bid) ? bid_occupied_ : ask_occupied_;
        PriceLevel& level = levels[o.level];

        if (o.prev != INVALID_SLOT) arena_[o.prev].next = o.next; 
        else level.head = o.next;
        if (o.next != INVALID_SLOT) arena_[o.next].prev = o.prev; 
        else level.tail = o.prev;

        level.total_qty -= o.qty;
        level.order_count -= 1;
        if (level.total_qty == 0) clear_bit(occupied, o.level);

        id_index_.erase(id);
        o.active = false;
        free_stack_[free_top_++] = s;
        return true;
    }

    // In-place quantity decrease only. A real venue resets price-time
    // priority on any increase, which means an increase is a new order,
    // not a modify — so an increase is rejected here.
    bool modify_qty(OrderId id, Qty new_qty) {
        Slot s = id_index_.find(id);
        if (s == INVALID_SLOT) return false;
        Order& o = arena_[s];
        if (new_qty > o.qty) return false;

        auto& levels   = (o.side == Side::Bid) ? bid_levels_ : ask_levels_;
        auto& occupied = (o.side == Side::Bid) ? bid_occupied_ : ask_occupied_;
        PriceLevel& level = levels[o.level];

        level.total_qty -= (o.qty - new_qty);
        o.qty = new_qty;
        if (level.total_qty == 0) clear_bit(occupied, o.level);
        return true;
    }

    bool best_bid(Price& out_price, Qty& out_qty) const {
        for (std::size_t w = WORDS; w-- > 0; ) {
            uint64_t word = bid_occupied_[w];
            if (word == 0) continue;
            
            int bit = 63 - std::countl_zero(word);
            std::size_t lvl = w * 64 + static_cast<std::size_t>(bit);
            out_price = base_price_ + static_cast<Price>(lvl);
            out_qty   = bid_levels_[lvl].total_qty;
            return true;
        }
        return false;
    }

    bool best_ask(Price& out_price, Qty& out_qty) const {
        for (std::size_t w = 0; w < WORDS; ++w) {
            uint64_t word = ask_occupied_[w];
            if (word == 0) continue;
            
            int bit = std::countr_zero(word);
            std::size_t lvl = w * 64 + static_cast<std::size_t>(bit);
            out_price = base_price_ + static_cast<Price>(lvl);
            out_qty   = ask_levels_[lvl].total_qty;
            return true;
        }
        return false;
    }
};

} 