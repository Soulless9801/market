import type { CSSProperties } from "react";
import { BOOK_DEPTH, type BookRow } from "./ViewModel";
import { formatPrice, formatQuantity } from "./format";

export function OrderBook({ asks, bids, clock }: { asks: BookRow[]; bids: BookRow[]; clock: number }) {
	return (
		<div className="order-book">
			<div className="order-book-heading">
				<div>
					<h2 className="market-panel-heading">Order book</h2>
					<div className="order-book-clock">Simulation time {clock}</div>
				</div>
				<dl className="order-book-quotes">
					<div><dt>Best bid</dt><dd>{formatPrice(bids[0]?.price ?? null)}</dd></div>
					<div><dt>Best ask</dt><dd>{formatPrice(asks[0]?.price ?? null)}</dd></div>
				</dl>
			</div>
			<div className="order-book-sides">
				<BookSide side="ASK" rows={asks} />
				<BookSide side="BID" rows={bids} />
			</div>
		</div>
	);
}

function BookSide({ side, rows }: { side: BookRow["side"]; rows: BookRow[] }) {
	return (
		<section className={`order-book-side order-book-side--${side.toLowerCase()}`} aria-label={`${side} price levels`}>
			<div className="order-book-side-heading"><h3>{side}</h3><span>Quantity</span></div>
			{/* Reserve the snapshot depth so additions/removals do not move the panels below. */}
			<div className="order-book-levels" style={{ "--book-level-count": BOOK_DEPTH } as CSSProperties}>
				{rows.length === 0 && <div className="order-book-empty">No resting orders</div>}
				{rows.map(row => (
					<div className="order-book-row" key={`${row.side}-${row.price}`}>
						<span className="order-book-price">{formatPrice(row.price)}</span>
						<span className="order-book-quantity">{formatQuantity(row.quantity)}</span>
						<div className="order-book-track" aria-hidden="true">
							<div className="order-book-fill" style={{ width: `${row.barWidth}%` }} />
						</div>
					</div>
				))}
			</div>
		</section>
	);
}
