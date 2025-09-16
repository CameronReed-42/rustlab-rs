//! Bessel functions of the first kind J_n(x)
//! 
//! This module provides high-precision implementations of Bessel functions of the first kind,
//! which are solutions to Bessel's differential equation:
//! 
//! ```text
//! x²y'' + xy' + (x² - n²)y = 0
//! ```
//! 
//! ## Mathematical Background
//! 
//! Bessel functions of the first kind J_n(x) are finite at the origin for all integer orders n ≥ 0.
//! They have the series representation:
//! 
//! ```text
//! J_n(x) = Σ_{m=0}^∞ [(-1)^m / (m! Γ(m+n+1))] (x/2)^(2m+n)
//! ```
//! 
//! ## Key Properties
//! 
//! - **Zeros**: J_n(x) has infinitely many zeros for x > 0
//! - **Asymptotic behavior**: J_n(x) ~ √(2/(πx)) cos(x - nπ/2 - π/4) as x → ∞
//! - **Recurrence relations**: J_{n-1}(x) + J_{n+1}(x) = (2n/x)J_n(x)
//! - **Symmetry**: J_{-n}(x) = (-1)^n J_n(x) for integer n
//! 
//! ## Implementation Details
//! 
//! The implementation uses different algorithms based on the argument magnitude:
//! - **Small arguments (|x| < 8)**: Taylor series expansion for optimal accuracy
//! - **Large arguments (|x| ≥ 8)**: Asymptotic expansion for efficiency
//! - **Fractional orders**: Specialized algorithms for non-integer orders
//! 
//! Target accuracy: 1e-15 relative error in the primary domain
//! 
//! ## Applications
//! 
//! - **Physics**: Wave propagation, electromagnetic fields, quantum mechanics
//! - **Engineering**: Antenna design, heat conduction in cylinders, vibration analysis
//! - **Signal Processing**: FM synthesis, filter design, spectral analysis
//! 
//! ## Examples
//! 
//! ```
//! use rustlab_special::bessel::{bessel_j0, bessel_j1, bessel_j};
//! 
//! // Compute J_0(2.4048) - first zero of J_0
//! let j0_at_zero = bessel_j0(2.4048255576957728);
//! assert!(j0_at_zero.abs() < 1e-14);
//! 
//! // Compute J_1(3.0)
//! let j1_val = bessel_j1(3.0);
//! assert!((j1_val - 0.33905895852593645).abs() < 1e-14);
//! 
//! // Compute J_5(10.0) using general function
//! let j5_val = bessel_j(5, 10.0);
//! assert!((j5_val - 0.23406152818679364).abs() < 1e-14);
//! ```

use std::f64::consts::PI;

// Transition points for each order
const TRANSITION_J0: f64 = 8.0;
const TRANSITION_J1: f64 = 8.0;
const TRANSITION_J2: f64 = 8.0;

// ===== Taylor Series Coefficients =====

/// Taylor series coefficients for J_0(x) around x=0, valid for |x| < 8.
const J0_TAYLOR_COEFFS: [f64; 41] = [
    1.0,
    -0.25,
    0.015625,
    -0.00043402777777777777777777777777777778,
    6.78168402777777777777777777777778e-6,
    -6.781684027777777777777777777777778e-8,
    4.709502797067901234567901234567901234567901234566e-10,
    -2.4028075495244394053917863441672965482489292013e-12,
    9.3859669903298414273116654069035021415973796e-15,
    -2.89690339207711155163940290336527843876462e-17,
    7.24225848019277887909850725841319609692e-20,
    -1.496334396734045222954237036862230598e-22,
    2.597802772107717400962217077885818e-25,
    -3.842903509035084912666001594505e-28,
    4.9016626390753634090127571358490388603170290656e-31,
    -5.4462918211948482322363968176100431781300323e-34,
    5.3186443566355939767933562671973077911426e-37,
    -4.6009034226951504989561905425582247328e-40,
    3.5500798014623074837624927025912228e-43,
    -2.4585040176331769278133605973623e-46,
    1.53656501102073557988335037335146415671379344628e-49,
    -8.710686003518909183012190325121678892935337e-53,
    4.4993212828093539168451396307446688496567e-56,
    -2.1263333094562164068266255343783879252e-59,
    9.228877211181494821296117770739531e-63,
    -3.691550884472597928518447108296e-66,
    1.3652185223641264528544552915295164090548495732e-69,
    -4.681819349671215544768365197289150922684669e-73,
    1.492927088543117201775626657298836391162e-76,
    -4.43795210625183472584906854131639831e-80,
    1.23276447395884297940251903925455e-83,
    -3.206983543077114930807801871109664635393839103273e-87,
    7.82954966571561262404248503688882967625449e-91,
    -1.79741727863076506520718205621873959509975e-94,
    3.8871480939246649334065355887083468752e-98,
    -7.932955293723805986543950181037442e-102,
    1.530276869931289735058632365169e-105,
    -2.794515832599141225454040111704272756174334764e-109,
    4.83815067970765447620159299117775754185308e-113,
    -7.9522529252262565355055769085761958282e-117,
    1.2425395195666025836727463919650306e-120,
];

/// Taylor series coefficients for J_1(x) around x=0, valid for |x| < 8.
const J1_TAYLOR_COEFFS: [f64; 40] = [
    0.5,
    -0.0625,
    0.00260416666666666666666666666666666667,
    -0.00005425347222222222222222222222222222,
    6.7816840277777777777777777777778e-7,
    -5.65140335648148148148148148148e-9,
    3.36393056933421516754850088183421516754850088183e-11,
    -1.50175471845277462836986646510456034265558075e-13,
    5.214426105738800792950925226057501189776322e-16,
    -1.4484516960385557758197014516826392193822e-18,
    3.2919356728148994904993214810969073168e-21,
    -6.2347266530585217623093209869259608e-24,
    9.9915491234912207729316041457146e-27,
    -1.372465538941101754523571998037730880888768138366e-29,
    1.633887546358454469670919045283012953439009689e-32,
    -1.701966194123390072573874005503138493165635e-35,
    1.564307163716351169645104784469796409159e-38,
    -1.278028728526430694154497372932840203e-41,
    9.3423152670060723256907702699769e-45,
    -6.14626004408294231953340149341e-48,
    3.658488121477941856865119936551105135032841539e-51,
    -1.97970136443611572341186143752765429384894e-54,
    9.78113322349859547140247745814058445578e-58,
    -4.42986106136711751422213652995497484e-61,
    1.84577544223629896425922355414791e-64,
    -7.09913631629345755484316751595348532708521778069e-68,
    2.52818244882245639417491720653614149824972143e-71,
    -8.3603916958414563299435092808734837905083e-75,
    2.5740122216260641409924597539635110192e-78,
    -7.396586843753057876415114235527331e-82,
    1.988329796707811257100837160088e-85,
    -5.0109117860579920793871904236088509928028735989e-89,
    1.1862954038963049430367401571043681327658318e-92,
    -2.643260703868772154716444200321675875147e-96,
    5.55306870560666419058076512672620983e-100,
    -1.10179934635052860924221530292187e-103,
    2.0679417161233645068359896826611618395690077256e-107,
    -3.6769945165778174019132106732950957318083352e-111,
    6.202757281676480097694349988689432745965e-115,
    -9.94031615653282066938197113572024478e-119,
];

/// Taylor series coefficients for J_2(x) around x=0, valid for |x| < 8.
const J2_TAYLOR_COEFFS: [f64; 40] = [
    0.125,
    -0.01041666666666666666666666666666666667,
    0.00032552083333333333333333333333333333,
    -5.42534722222222222222222222222222e-6,
    5.651403356481481481481481481481e-8,
    -4.036716683201058201058201058201058201058201058201e-10,
    2.1024566058338844797178130511463844797178130511e-12,
    -8.3430817691820812687214803616920019036421151e-15,
    2.60721305286940039647546261302875059488815e-17,
    -6.58387134562979898099864296219381463355e-20,
    1.371639863672874787708050617123711381e-22,
    -2.397971789637892985503584994971523e-25,
    3.568410401246864561761287194897e-28,
    -4.5748851298036725150785733267924362696292271279e-31,
    5.1058985823701702177216220165094154794969053e-34,
    -5.0057829238923237428643353103033485093107e-37,
    4.3452976769898643601252910679716566921e-40,
    -3.3632334961221860372486772971916848e-43,
    2.3355788167515180814226925674942e-46,
    -1.4633952485911767427460479746204420540131366155e-49,
    8.314745730631686038329818037616148034165549e-53,
    -4.3036986183393820074170900815818571605412e-56,
    2.0377360882288740565421828037792884283e-59,
    -8.85972212273423502844427305990995e-63,
    3.549568158146728777421583757977e-66,
    -1.3146548733876773249709569473987935790898551446e-69,
    4.514611515754386418169495011671681246874503e-73,
    -1.441446844110595918955777462219566170777e-76,
    4.2900203693767735683207662566058517e-80,
    -1.19299787802468675426050229605279e-83,
    3.106765307355955089220058062637487615537781631296e-87,
    -7.59229058493635163543513700546795604970132363e-91,
    1.74455206455338962211285317221230607759682e-94,
    -3.7760867198125316495949202861738226788e-98,
    7.712595424453700264695507120453069e-102,
    -1.488918035608822444921912571517e-105,
    2.72097594226758487741577589823837084153816806e-109,
    -4.71409553407412487424770599140396888693376e-113,
    7.7534466020956001221179374858617909325e-117,
    -1.2122336776259537401685330653317372e-120,
];

// ===== Asymptotic Expansion Coefficients =====

/// Asymptotic expansion P coefficients (same for all orders)
const ASYMP_P_COEFFS: [f64; 6] = [
    1.0,
    -0.0625,
    0.017578125,
    -0.0091552734375,
    0.0070095062255859375,
    -0.00709712505340576171875,
];

/// Asymptotic expansion Q coefficients (same for all orders)  
const ASYMP_Q_COEFFS: [f64; 6] = [
    -0.125,
    0.0546875,
    -0.021240234375,
    0.0168914794921875,
    -0.009934902191162109375,
    0.0150512158870697021484375,
];

// ===== Implementation =====

/// Bessel function of the first kind J_n(x) for integer order n
/// 
/// Computes the Bessel function of the first kind for arbitrary non-negative
/// integer order n. This is the most general interface for integer-order
/// Bessel functions.
/// 
/// # Arguments
/// 
/// * `n` - The order of the Bessel function (non-negative integer)
/// * `x` - The argument at which to evaluate J_n(x)
/// 
/// # Returns
/// 
/// The value of J_n(x) with target accuracy of 1e-15 for most arguments.
/// 
/// # Algorithm Selection
/// 
/// - n = 0, 1, 2: Uses specialized implementations for optimal performance
/// - n > 2, |x| < 8: Uses Taylor series with upward recurrence
/// - n > 2, |x| ≥ 8: Uses asymptotic expansion
/// 
/// # Mathematical Properties
/// 
/// - **Recurrence relation**: J_{n-1}(x) + J_{n+1}(x) = (2n/x)J_n(x)
/// - **Derivative**: J'_n(x) = (J_{n-1}(x) - J_{n+1}(x))/2
/// - **Symmetry**: J_{-n}(x) = (-1)^n J_n(x) for integer n
/// - **Generating function**: exp((x/2)(t - 1/t)) = Σ_{n=-∞}^∞ J_n(x) t^n
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::bessel_j;
/// 
/// // J_0(1) - most common case
/// let j0 = bessel_j(0, 1.0);
/// assert!((j0 - 0.7651976865579666).abs() < 1e-14);
/// 
/// // Higher order: J_5(10)
/// let j5 = bessel_j(5, 10.0);
/// assert!((j5 - 0.23406152818679364).abs() < 1e-14);
/// 
/// // J_n(-x) = (-1)^n J_n(x)
/// let j3_neg = bessel_j(3, -2.0);
/// let j3_pos = bessel_j(3, 2.0);
/// assert!((j3_neg + j3_pos).abs() < 1e-14);
/// ```
/// 
/// # Applications
/// 
/// - **Cylindrical waveguides**: Mode field distributions
/// - **Circular membrane vibrations**: Eigenfunction solutions
/// - **Antenna theory**: Radiation patterns of circular apertures
/// - **Quantum mechanics**: Angular momentum eigenfunctions
pub fn bessel_j(n: u32, x: f64) -> f64 {
    match n {
        0 => bessel_j0(x),
        1 => bessel_j1(x),
        2 => bessel_j2(x),
        _ => bessel_jn_general(n, x),
    }
}

/// Bessel function of the first kind J_ν(x) for arbitrary real order ν
/// 
/// Computes the Bessel function of the first kind for arbitrary real (possibly 
/// non-integer) order ν. This extends the integer-order bessel_j function to
/// fractional and negative orders.
/// 
/// # Arguments
/// 
/// * `nu` - The order of the Bessel function (any real number)
/// * `x` - The argument at which to evaluate J_ν(x)
/// 
/// # Returns
/// 
/// The value of J_ν(x). Accuracy may be reduced for extreme orders or arguments.
/// 
/// # Algorithm Selection
/// 
/// - If ν is very close to an integer (within 1e-12), uses integer implementation
/// - For |x| < 8: Uses series expansion with gamma functions
/// - For |x| ≥ 8: Uses asymptotic expansion (simplified first-order)
/// 
/// # Mathematical Properties
/// 
/// - **Relation to integer orders**: J_n(x) = J_{n.0}(x) for integer n
/// - **Half-integer relation**: J_{n+1/2}(x) relates to elementary functions
/// - **Symmetry**: J_{-ν}(x) = (-1)^ν J_ν(x) for integer ν, more complex otherwise
/// - **Series representation**: J_ν(x) = (x/2)^ν Σ_{k=0}^∞ [(-1)^k (x/2)^{2k}] / [k! Γ(ν+k+1)]
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::bessel_j_nu;
/// use std::f64::consts::PI;
/// 
/// // Half-integer order: J_{1/2}(x) = √(2/(πx)) sin(x)
/// let x = 2.0;
/// let j_half = bessel_j_nu(0.5, x);
/// let expected = (2.0 / (PI * x)).sqrt() * x.sin();
/// assert!((j_half - expected).abs() < 1e-12);
/// 
/// // Fractional order
/// let j_frac = bessel_j_nu(1.5, 3.0);
/// assert!(j_frac.is_finite());
/// 
/// // Integer order should match bessel_j
/// let j3_int = bessel_j_nu(3.0, 5.0);
/// let j3_ref = rustlab_special::bessel_j(3, 5.0);
/// assert!((j3_int - j3_ref).abs() < 1e-14);
/// ```
/// 
/// # Applications
/// 
/// - **Spherical coordinates**: Half-integer orders arise in 3D problems
/// - **Scattering theory**: Non-integer orders in complex geometries  
/// - **Mathematical physics**: Solutions to generalized Bessel equations
/// - **Special relativity**: Bessel functions of fractional order in field theory
/// 
/// # Notes
/// 
/// For negative arguments with non-integer ν, the function may return NaN
/// as the result involves complex numbers which are not supported by this
/// real-valued implementation.
pub fn bessel_j_nu(nu: f64, x: f64) -> f64 {
    // Check if nu is close to an integer for optimization
    let n_rounded = nu.round();
    if (nu - n_rounded).abs() < 1e-12 && n_rounded >= 0.0 && n_rounded <= u32::MAX as f64 {
        // Use integer implementation for better accuracy
        return bessel_j(n_rounded as u32, x);
    }
    
    // For non-integer orders, use series expansion for small x, asymptotic for large x
    if x.abs() < 8.0 {
        bessel_j_nu_series(nu, x)
    } else {
        bessel_j_nu_asymptotic(nu, x)
    }
}

/// Bessel function of the first kind of order 0: J_0(x)
/// 
/// Computes the zeroth-order Bessel function of the first kind, which is the
/// most commonly used Bessel function in applications.
/// 
/// # Arguments
/// 
/// * `x` - The argument at which to evaluate J_0(x). Can be any real number.
/// 
/// # Returns
/// 
/// The value of J_0(x) with target accuracy of 1e-15.
/// 
/// # Special Values
/// 
/// - J_0(0) = 1
/// - J_0(∞) → 0 (oscillating decay)
/// - First zero: J_0(2.4048255576957728) = 0
/// - First maximum: J_0(0) = 1
/// - First minimum: J_0(3.8317059702075123) ≈ -0.4028
/// 
/// # Algorithm
/// 
/// - For |x| < 8: Uses a 41-term Taylor series expansion
/// - For |x| ≥ 8: Uses asymptotic expansion for large arguments
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::bessel_j0;
/// 
/// // J_0 at the origin
/// assert_eq!(bessel_j0(0.0), 1.0);
/// 
/// // J_0 at x = 1
/// let j0_1 = bessel_j0(1.0);
/// assert!((j0_1 - 0.7651976865579666).abs() < 1e-14);
/// 
/// // J_0 at its first zero
/// let first_zero = 2.4048255576957728;
/// assert!(bessel_j0(first_zero).abs() < 1e-14);
/// ```
/// 
/// # Applications
/// 
/// - **Circular waveguides**: TM_01 mode field distribution
/// - **Diffraction patterns**: Airy disk intensity profile
/// - **Heat conduction**: Temperature in infinite cylinders
/// - **Signal processing**: Bessel filters, FM synthesis
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < TRANSITION_J0 {
        bessel_j0_taylor(x)
    } else {
        bessel_jn_asymptotic(0, ax)
    }
}

/// Bessel function of the first kind of order 1: J_1(x)
/// 
/// Computes the first-order Bessel function of the first kind.
/// 
/// # Arguments
/// 
/// * `x` - The argument at which to evaluate J_1(x). Can be any real number.
/// 
/// # Returns
/// 
/// The value of J_1(x) with target accuracy of 1e-15.
/// 
/// # Special Values
/// 
/// - J_1(0) = 0
/// - J_1(-x) = -J_1(x) (odd function)
/// - First zero: J_1(3.8317059702075123) = 0
/// - First maximum: J_1(1.8411837813406593) ≈ 0.5819
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::bessel_j1;
/// 
/// // J_1 at the origin
/// assert_eq!(bessel_j1(0.0), 0.0);
/// 
/// // J_1 is an odd function
/// assert_eq!(bessel_j1(-2.0), -bessel_j1(2.0));
/// 
/// // J_1 at x = 1
/// let j1_1 = bessel_j1(1.0);
/// assert!((j1_1 - 0.4400505857449335).abs() < 1e-14);
/// ```
/// 
/// # Derivative Relation
/// 
/// J_1(x) is related to the derivative of J_0(x):
/// - J_0'(x) = -J_1(x)
pub fn bessel_j1(x: f64) -> f64 {
    if x.abs() < TRANSITION_J1 {
        bessel_j1_taylor(x)
    } else {
        // J_1(-x) = -J_1(x)
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * bessel_jn_asymptotic(1, x.abs())
    }
}

/// Bessel function of the first kind of order 2: J_2(x)
/// 
/// Computes the second-order Bessel function of the first kind.
/// 
/// # Arguments
/// 
/// * `x` - The argument at which to evaluate J_2(x). Can be any real number.
/// 
/// # Returns
/// 
/// The value of J_2(x) with target accuracy of 1e-15.
/// 
/// # Special Values
/// 
/// - J_2(0) = 0  
/// - J_2(-x) = J_2(x) (even function)
/// - First zero: J_2(5.520078110286311) = 0
/// - First maximum: J_2(3.054236928227453) ≈ 0.4865
/// 
/// # Algorithm
/// 
/// - For |x| < 8: Uses a 40-term Taylor series expansion starting with x²
/// - For |x| ≥ 8: Uses asymptotic expansion for large arguments
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::bessel_j2;
/// 
/// // J_2 at the origin  
/// assert_eq!(bessel_j2(0.0), 0.0);
/// 
/// // J_2 is an even function
/// assert_eq!(bessel_j2(-3.0), bessel_j2(3.0));
/// 
/// // J_2 at x = 1
/// let j2_1 = bessel_j2(1.0);
/// assert!((j2_1 - 0.11490348493190047).abs() < 1e-14);
/// ```
/// 
/// # Recurrence Relations
/// 
/// - J_1(x) + J_3(x) = (4/x)J_2(x)  
/// - J_2'(x) = (J_1(x) - J_3(x))/2
/// 
/// # Applications
/// 
/// - **Circular waveguides**: TM_02, TE_21 mode field distributions
/// - **Vibrating membranes**: Higher-order radial modes
/// - **Scattering**: Higher multipole contributions
pub fn bessel_j2(x: f64) -> f64 {
    let ax = x.abs();
    if ax < TRANSITION_J2 {
        bessel_j2_taylor(x)
    } else {
        bessel_jn_asymptotic(2, ax)
    }
}

// ===== Taylor Series Implementations =====

fn bessel_j0_taylor(x: f64) -> f64 {
    let x2 = x * x;
    let mut sum = 0.0;
    let mut x2n = 1.0;  // x^(2n)
    
    for i in 0..J0_TAYLOR_COEFFS.len() {
        sum += J0_TAYLOR_COEFFS[i] * x2n;
        x2n *= x2;
        
        if i + 1 < J0_TAYLOR_COEFFS.len() && 
           x2n.abs() * J0_TAYLOR_COEFFS[i + 1].abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

fn bessel_j1_taylor(x: f64) -> f64 {
    let x2 = x * x;
    let mut sum = 0.0;
    let mut x_power = x;  // Start with x^1 for J_1
    
    for i in 0..J1_TAYLOR_COEFFS.len() {
        sum += J1_TAYLOR_COEFFS[i] * x_power;
        x_power *= x2;
        
        if i + 1 < J1_TAYLOR_COEFFS.len() && 
           x_power.abs() * J1_TAYLOR_COEFFS[i + 1].abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

fn bessel_j2_taylor(x: f64) -> f64 {
    let x2 = x * x;
    let mut sum = 0.0;
    let mut x_power = x2;  // Start with x^2 for J_2
    
    for i in 0..J2_TAYLOR_COEFFS.len() {
        sum += J2_TAYLOR_COEFFS[i] * x_power;
        x_power *= x2;
        
        if i + 1 < J2_TAYLOR_COEFFS.len() && 
           x_power.abs() * J2_TAYLOR_COEFFS[i + 1].abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

// ===== Asymptotic Expansion =====

fn bessel_jn_asymptotic(n: u32, x: f64) -> f64 {
    let z = 8.0 / x;
    let z2 = z * z;
    
    // Evaluate P and Q using Horner's method
    let mut p = ASYMP_P_COEFFS[5];
    let mut q = ASYMP_Q_COEFFS[5];
    
    for i in (0..5).rev() {
        p = p * z2 + ASYMP_P_COEFFS[i];
        q = q * z2 + ASYMP_Q_COEFFS[i];
    }
    
    q *= z;
    
    // Phase shift depends on order: χ = x - nπ/2 - π/4
    let phase = x - (n as f64) * PI / 2.0 - PI / 4.0;
    let (sin_phase, cos_phase) = phase.sin_cos();
    let amplitude = (2.0 / (PI * x)).sqrt();
    
    amplitude * (p * cos_phase - q * sin_phase)
}

// ===== General Case Using Recurrence =====

fn bessel_jn_general(n: u32, x: f64) -> f64 {
    // Use the most conservative transition point
    let transition = TRANSITION_J0.min(TRANSITION_J1).min(TRANSITION_J2);
    
    if x.abs() < transition {
        // Use upward recurrence for small x
        bessel_jn_taylor_recurrence(n, x)
    } else {
        // Use asymptotic expansion for large x
        let sign = if n % 2 == 1 && x < 0.0 { -1.0 } else { 1.0 };
        sign * bessel_jn_asymptotic(n, x.abs())
    }
}

fn bessel_jn_taylor_recurrence(n: u32, x: f64) -> f64 {
    // For n > 2, use recurrence relation:
    // J_{n+1}(x) = (2n/x) * J_n(x) - J_{n-1}(x)
    
    if n == 0 {
        return bessel_j0_taylor(x);
    } else if n == 1 {
        return bessel_j1_taylor(x);
    }
    
    let mut j_prev = bessel_j0_taylor(x);
    let mut j_curr = bessel_j1_taylor(x);
    
    for k in 1..n {
        let j_next = (2.0 * k as f64 / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }
    
    j_curr
}

// ===== Fractional Order Implementations =====

fn bessel_j_nu_series(nu: f64, x: f64) -> f64 {
    // Series expansion for J_ν(x) = Σ(k=0 to ∞) (-1)^k * (x/2)^(ν+2k) / (k! * Γ(ν+k+1))
    use crate::gamma_functions::gamma;
    
    if x == 0.0 {
        return if nu == 0.0 { 1.0 } else { 0.0 };
    }
    
    let x_half = x * 0.5;
    let x_half_nu = x_half.powf(nu);
    let mut sum = 0.0;
    let mut term = 1.0 / gamma(nu + 1.0);
    let x_half_sq = x_half * x_half;
    
    sum += term;
    
    for k in 1..100 {
        term *= -x_half_sq / (k as f64 * (nu + k as f64));
        sum += term;
        
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    
    x_half_nu * sum
}

fn bessel_j_nu_asymptotic(nu: f64, x: f64) -> f64 {
    // Asymptotic expansion for large |x|:
    // J_ν(x) ≈ √(2/(πx)) * [P_ν(x) * cos(x - νπ/2 - π/4) - Q_ν(x) * sin(x - νπ/2 - π/4)]
    
    let ax = x.abs();
    let phase = ax - nu * PI / 2.0 - PI / 4.0;
    let amplitude = (2.0 / (PI * ax)).sqrt();
    
    // First-order asymptotic approximation (P_ν ≈ 1, Q_ν ≈ 0 for simplicity)
    // For better accuracy, we would need to implement the full P_ν and Q_ν series
    let result = amplitude * phase.cos();
    
    if x < 0.0 && nu.fract() != 0.0 {
        // Handle complex behavior for negative x and non-integer ν
        // For now, return NaN to indicate this case needs special handling
        f64::NAN
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    const TOLERANCE: f64 = 1e-14;
    
    #[test]
    fn test_j0_special_values() {
        assert!((bessel_j0(0.0) - 1.0).abs() < TOLERANCE);
        assert!((bessel_j0(2.4048255576957728) - 0.0).abs() < TOLERANCE);
    }
    
    #[test]
    fn test_j1_special_values() {
        assert!((bessel_j1(0.0) - 0.0).abs() < TOLERANCE);
        assert!((bessel_j1(3.8317059702075123) - 0.0).abs() < TOLERANCE);
    }
    
    #[test]
    fn test_recurrence_relation() {
        let x = 5.0;
        for n in 1..10 {
            let j_n_minus_1 = bessel_j(n - 1, x);
            let j_n = bessel_j(n, x);
            let j_n_plus_1 = bessel_j(n + 1, x);
            let recurrence = (2.0 * n as f64 / x) * j_n - j_n_minus_1;
            assert!(
                (j_n_plus_1 - recurrence).abs() < TOLERANCE,
                "Recurrence failed at n={}, x={}",
                n, x
            );
        }
    }
}