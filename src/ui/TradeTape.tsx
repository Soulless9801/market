import type { CSSProperties } from "react";
import { TRADE_TAPE_LIMIT, type TradeTapeEntry } from "./ViewModel";
import { formatPrice, formatQuantity } from "./format";

export function TradeTape({ trades }: { trades: TradeTapeEntry[] }) {
	return (
		<div className="trade-tape">
			<h2 className="market-panel-heading">Trade tape</h2>
			<div className="trade-tape-columns trade-tape-heading" aria-hidden="true">
				<span>Time</span><span>Side</span><span>Quantity</span><span>Price</span>
			</div>
			{/* Keep the same footprint from an empty tape to a full, constantly rolling tape. */}
			<ol className="trade-tape-entries" aria-label="Recent trades" style={{ "--trade-tape-limit": TRADE_TAPE_LIMIT } as CSSProperties}>
				{trades.map(trade => (
					<li className="trade-tape-columns trade-tape-row" key={trade.id}>
						<span title={`Time ${trade.timestamp}`}>{trade.timestamp}</span>
						<span>{trade.side}</span>
						<span title={`Quantity ${formatQuantity(trade.quantity)}`}>{formatQuantity(trade.quantity)}</span>
						<span title={formatPrice(trade.price)}>{formatPrice(trade.price)}</span>
					</li>
				))}
			</ol>
			{trades.length === 0 && <p className="trade-tape-empty">No trades yet</p>}
		</div>
	);
}
