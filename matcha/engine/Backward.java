package matcha.engine;

/**
 * Single Abstract Method (SAM)/Functional interface for the derivative of a
 * function.
 */
public interface Backward {
    void pass();
}
