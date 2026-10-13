// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

pub struct PrivateTraitMarker;

macro_rules! private_trait {
	() => {
		#[doc(hidden)]
		fn _private() -> crate::private::PrivateTraitMarker;
	};
}
macro_rules! private_impl {
	() => {
		fn _private() -> crate::private::PrivateTraitMarker {
			crate::private::PrivateTraitMarker
		}
	};
}
