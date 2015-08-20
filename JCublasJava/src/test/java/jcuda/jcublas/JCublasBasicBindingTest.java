package jcuda.jcublas;
/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */



import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCublas and JCublas2 class 
 */
public class JCublasBasicBindingTest
{
    public static void main(String[] args)
    {
        JCublasBasicBindingTest test = new JCublasBasicBindingTest();
        test.testJCublas();
        test.testJCublas2();
    }

    @Test
    public void testJCublas()
    {
        assertTrue(BasicBindingTest.testBinding(JCublas.class));
    }
    
    @Test
    public void testJCublas2()
    {
        assertTrue(BasicBindingTest.testBinding(JCublas2.class));
    }

}
