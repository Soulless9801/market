// Built-in layout metadata. New layouts register their own descriptor and builder.
// The historical "cnn" layout is just an input contract; it does not enable a
// Python CNN architecture. CNN training is still deferred, and an MLP can use it.
const mlpNames = [
	"log_midprice_over_reference",
	"log_history_max_over_reference",
	"log_history_min_over_reference",
	"log_midprice_over_last",
	"log_midprice_over_middle_observation",
	"log_midprice_over_first",
	"spread_over_midprice",
	"book_bid_volume_fraction",
	"recent_trade_count_imbalance",
	"log1p_rms_log_return",
	"top_level_quantity_imbalance",
	"log_best_bid_over_midprice",
	"log_best_ask_over_midprice",
	"log1p_best_bid_quantity",
	"log1p_best_ask_quantity",
	...Array.from(
		{ length: 20 },
		(_, i) => `chronological_log_return_${i}`,
	),
	...Array.from({ length: 5 }, (_, i) => `depth_quantity_imbalance_${i}`),
];
export const MLP_FEATURE_SCHEMA = {
	version: 1,
	names: mlpNames,
	minimumWarmupSteps: 20,
	priceHistoryLimit: 20,
	tradeHistoryLimit: 20,
	bookDepth: 10,
	historyOrder: "oldest-to-newest",
	padding: "none-require-20-history-observations",
	description:
		"15 market summaries, 20 chronological log returns (last ends at current midprice), 5 signed depth imbalances; middle price is chronological, volatility is RMS return; invalid log-price ratios and non-finite summaries become zero.",
};
export const CNN_FEATURE_SCHEMA = {
	version: 1,
	names: Array.from({ length: 30 }, (_, i) => `midprice_window_${i}`),
	minimumWarmupSteps: 0,
	priceHistoryLimit: 20,
	tradeHistoryLimit: 20,
	bookDepth: 10,
	historyOrder: "oldest-to-newest",
	padding: "prepend-current-midprice-to-length-30",
	description:
		"30 price levels: up to 20 historical pre-order midprices, left-padded with the current midprice; padding is not additional historical data.",
};
