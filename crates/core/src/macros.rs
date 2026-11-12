//! Crate-internal macros.
//!
//! Declared first in `lib.rs` with `#[macro_use]` so that every module can use them: a
//! `macro_rules!` definition is only visible textually, and one living inside `domain`
//! would be invisible to `journal`.

/// Declare a closed state enum with a single source of truth for its names.
///
/// The textual name appears three times per variant in the obvious hand-written version
/// (`Display`, `FromStr`, the serde attribute), and a mismatch between them is exactly the
/// bug that turns a recoverable journal into a `blocked` task: the writer says `running`,
/// the reader expects `Running`, and recovery declares the state unknown. One table, one
/// spelling.
///
/// Also generates `ALL`, which is what makes the transition-table property test exhaustive
/// instead of a list that quietly rots when a variant is added.
macro_rules! closed_state {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The name used in journals, state files and command output.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }

            /// Parse the name written by [`Self::as_str`].
            pub fn parse(text: &str) -> ::std::result::Result<Self, $crate::domain::UnknownState> {
                match text {
                    $($text => Ok(Self::$variant),)+
                    other => Err($crate::domain::UnknownState::new(stringify!($name), other)),
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        // Hand-written rather than derived so that the wire format is the same table as
        // everything else. A derived `Deserialize` would report an opaque "unknown variant"
        // without naming the closed set, and would silently accept nothing else useful.
        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                let text = <::std::borrow::Cow<'de, str> as ::serde::Deserialize>::deserialize(
                    deserializer,
                )?;
                Self::parse(&text).map_err(::serde::de::Error::custom)
            }
        }
    };
}
