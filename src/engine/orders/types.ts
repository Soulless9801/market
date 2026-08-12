export type Side = "BUY" | "SELL";

export type OrderType = "LIMIT" | "MARKET";

export type OrderStatus =
	"FILLED" | "PARTIALLY_FILLED" | "RESTING" | "UNFILLED" | "REJECTED";

export interface NewOrderRequestBase {
	id?: string;
	participantId: string;
	side: Side;
	quantity: number;
}

export interface NewLimitOrderRequest extends NewOrderRequestBase {
	type: "LIMIT";
	price: number;
}

export interface NewMarketOrderRequest extends NewOrderRequestBase {
	type: "MARKET";
}

export type NewOrderRequest = NewLimitOrderRequest | NewMarketOrderRequest;

export interface IncomingOrder extends Omit<NewOrderRequestBase, "id"> {
	id: string;
	timestamp: number;
	type: OrderType;
	price?: number;
}

export interface RestingOrder {
	id: string;
	participantId: string;
	side: Side;
	price: number;
	initialQuantity: number;
	remainingQuantity: number;
	timestamp: number;
}

export interface TradeEvent {
	tradeId: string;
	price: number;
	quantity: number;
	buyOrderId: string;
	buyerParticipantId: string;
	sellOrderId: string;
	sellerParticipantId: string;
	makerOrderId: string;
	makerParticipantId: string;
	takerOrderId: string;
	takerParticipantId: string;
	aggressorSide: Side;
	timestamp: number;
}

export interface ExecutionReport {
	orderId: string;
	status: OrderStatus;
	filledQuantity: number;
	remainingQuantity: number;
	wasAddedToBook: boolean;
	trades: TradeEvent[];
	rejectionReason?: string;
}

export interface CancelResult {
	orderId: string;
	cancelled: boolean;
	cancelledQuantity: number;
	reason?: string;
}

export interface BookLevelSnapshot {
	price: number;
	quantity: number;
	orderCount: number;
}

export interface OrderBookSnapshot {
	bids: BookLevelSnapshot[];
	asks: BookLevelSnapshot[];
}
